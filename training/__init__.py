"""The training code every method shares, on any environment: the dataset
(`data`), the agent loop and its bash tool (`loop`), the yes/no grader
(`grader`), the logging (`progress`, `render`), the launcher (`run`) and
the verl recipe (`train.yaml`, `tools.yaml`). The methods live in
`methods/`, the environments in `envs/`.
"""
