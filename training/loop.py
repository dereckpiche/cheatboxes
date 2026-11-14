"""The agent loop that plays one environment episode per trajectory, and its
one tool.

verl calls `await loop.run(sampling_params, **row)` once per trajectory,
on an event loop shared by hundreds of them (so anything blocking goes
through `asyncio.to_thread`), and `await tool.execute(...)` once per tool
call. Tokens, masks and the policy update stay verl's: `EpisodeLoop` is
verl's `ToolAgentLoop` with the episode opened before, graded after, and
the episode's output budget, if it has one, enforced in between. Calls are
`envs.commons.tool`'s, read by `training.parsers`; the chat template gets
no tool list, so the prompt carries none of its tool preamble. The
environment is whichever the row names (see `envs`).
"""

import asyncio
import json
import os
import uuid

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentState,
    ToolAgentLoop,
)
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from training import instances
from training.parsers import BashTagParser

REGISTERED_PARSERS = (BashTagParser,)
"""Tool-call parsers this module registers with verl by importing them."""
THINK_TAGS = (("<think>", "</think>"),)
"""The reasoning blocks models write."""

EPISODES: dict[str, object] = {}
"""This worker's live episodes, keyed by an id the loop hands to the tool
through verl's per-trajectory `tools_kwargs`."""
CALLS: dict[str, int] = {}
"""The tool calls each live episode has received."""
CLOSING_TOKENS = 256
"""What the model may write in an episode's closing turn."""
CLOSING_ROOM = 512
"""The response length an episode must have left for its closing turn:
the closing message and an answer to it."""

SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command."}
                },
                "required": ["command"],
            },
        },
    }
)


class Bash(BaseTool):
    """The model's one tool: a command run in its episode's sandbox."""

    def __init__(self, config: dict, tool_schema=None):
        """verl builds one per worker; TOOL_PARALLEL caps concurrent calls."""
        super().__init__(config, tool_schema or SCHEMA)
        self.slots = asyncio.Semaphore(int(os.environ.get("TOOL_PARALLEL", 8)))

    async def execute(self, instance_id, parameters, agent_data=None, **_):
        """Run the command; the output is what the model sees next."""
        key = agent_data.tools_kwargs["episode"]
        episode = EPISODES[key]
        CALLS[key] += 1
        async with self.slots:
            output = await asyncio.to_thread(
                episode.step, str(parameters.get("command", ""))
            )
        return ToolResponse(text=output), 0.0, {}


class EpisodeLoop(ToolAgentLoop):
    """verl's tool loop around one environment episode."""

    def __init__(self, *args, **kwargs):
        """verl's loop, with no tool list for the chat template to render."""
        super().__init__(*args, **kwargs)
        self.tool_schemas = []

    async def run(self, sampling_params, **row):
        """Open the episode, let verl run the conversation, grade it, log
        it.
        """
        split = row["extra_info"]["split"]
        key = uuid.uuid4().hex
        episode = await asyncio.to_thread(instances.episode, row["extra_info"])
        EPISODES[key], CALLS[key] = episode, 0
        try:
            row = {**row, "tools_kwargs": {"episode": key}}
            output = await super().run(sampling_params, **row)
            reward, metrics = await asyncio.to_thread(episode.grade)
        finally:
            EPISODES.pop(key, None)
            calls = CALLS.pop(key, 0)
            await asyncio.to_thread(episode.close)
        output.reward_score = reward
        record = self.record(output, split, reward, metrics, calls)
        record["env"] = row["extra_info"]["env"]
        record["task"] = row["extra_info"]["index"]
        ids = output.prompt_ids + output.response_ids
        record["transcript"] = self.tokenizer.decode(ids)
        append(record)
        return output

    async def _handle_generating_state(self, agent_data, params, **kwargs):
        """Each model turn: stop once the episode is over; once its turns
        or budget, where it has them, are spent, or verl ends it (a turn
        cut off or without a tool call), give it its closing turn if it has
        one; otherwise cap the turn at what remains of the budget. The
        response mask is 1 on model tokens, so its sum is what the model
        has spent.
        """
        episode = EPISODES[agent_data.tools_kwargs["episode"]]
        if episode.done:
            return AgentState.TERMINATED
        turns = getattr(episode, "turns", None)
        budget = getattr(episode, "budget", None)
        used = sum(agent_data.response_mask)
        spent = (turns is not None and agent_data.assistant_turns >= turns) or (
            budget is not None and used >= budget
        )
        if spent:
            return await self.last_turn(agent_data, episode, params, **kwargs)
        if budget is not None:
            params = {**params, "max_tokens": budget - used}
        state = await super()._handle_generating_state(
            agent_data, params, **kwargs
        )
        if state == AgentState.TERMINATED and not episode.done:
            return await self.last_turn(agent_data, episode, params, **kwargs)
        return state

    async def last_turn(self, agent_data, episode, params, **kwargs):
        """The episode's next closing turn, if it gives one and has
        CLOSING_ROOM of response length left: the model answers its closing
        message within CLOSING_TOKENS, or what is left if less.
        """
        room = self.response_length - len(agent_data.response_mask)
        if room < CLOSING_ROOM or not await self.close(agent_data, episode):
            return AgentState.TERMINATED
        room = self.response_length - len(agent_data.response_mask)
        if room <= 0:
            return AgentState.TERMINATED
        params = {**params, "max_tokens": min(CLOSING_TOKENS, room)}
        return await super()._handle_generating_state(
            agent_data, params, **kwargs
        )

    async def close(self, agent_data, episode) -> bool:
        """Append the episode's next closing message as a user turn the
        model answers within CLOSING_TOKENS; False if it has no `closing`
        or it returns None (it has no more to give).
        """
        closing = getattr(episode, "closing", None)
        if closing is None:
            return False
        text = await asyncio.to_thread(closing)
        if text is None:
            return False
        previous = list(agent_data.messages)
        agent_data.messages.append({"role": "user", "content": text})
        merged, mask, logprobs = await self.ct_merge_context_msg(
            previous,
            agent_data.messages,
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs or None,
            tools=getattr(
                agent_data, "_active_tool_schemas", self.tool_schemas
            ),
        )
        agent_data.prompt_ids, agent_data.response_mask = merged.token_ids, mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = logprobs or []
        self.skip_thinking(agent_data)
        return True

    def skip_thinking(self, agent_data) -> None:
        """Close the reasoning block the generation prompt opens, as the
        chat template does with thinking off, so that the short closing turn
        is an answer, not the start of more reasoning. The added tokens are
        context (mask 0), not the model's.
        """
        tail = self.tokenizer.decode(agent_data.prompt_ids[-4:]).rstrip()
        opened = [close for start, close in THINK_TAGS if tail.endswith(start)]
        text = f"\n{opened[0]}\n\n" if opened else "<think>\n\n</think>\n\n"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        agent_data.prompt_ids = agent_data.prompt_ids + ids
        agent_data.response_mask = agent_data.response_mask + [0] * len(ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(ids)

    def record(
        self, output, split: str, reward: float, metrics: dict, calls: int
    ) -> dict:
        """What the episode log keeps about the episode: its grade
        (reward_task, the reward trained on), the environment's metrics,
        tool calls and token counts.
        """
        mask = output.response_mask
        ids = output.prompt_ids + output.response_ids
        generated = [0] * len(output.prompt_ids) + mask
        return {
            "split": split,
            "reward_task": reward,
            "metrics": metrics,
            "tool_calls": calls,
            "output_tokens": sum(mask),
            "reasoning_tokens": thinking(self.tokenizer, ids, generated),
            "input_tokens": len(ids) - sum(mask),
        }


def append(record: dict) -> None:
    """Append one episode's record to EPISODE_LOG, if set."""
    path = os.environ.get("EPISODE_LOG")
    if path:
        with open(path, "a") as file:
            file.write(json.dumps(record) + "\n")


def think_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """The ids of the tokenizer's reasoning tags: openers, then closers.
    Tags it does not hold as single tokens are left out.
    """
    vocabulary = tokenizer.get_vocab()
    starts = {vocabulary[s] for s, _ in THINK_TAGS if s in vocabulary}
    ends = {vocabulary[e] for _, e in THINK_TAGS if e in vocabulary}
    return starts, ends


def thinking(tokenizer, ids: list[int], mask: list[int]) -> int:
    """Model tokens inside a reasoning block, between an opening tag (which
    may come from the prompt's template) and its closing tag.
    """
    starts, ends = think_token_ids(tokenizer)
    inside, count = False, 0
    for token, generated in zip(ids, mask):
        if token in starts or token in ends:
            inside = token in starts
        elif inside and generated:
            count += 1
    return count
