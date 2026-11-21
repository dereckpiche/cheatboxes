// The JavaScript counterpart of harness.py. Runs inside the grading
// sandbox: load solution.js, evaluate each call from calls.json, write
// results.json (each call's result as JSON, null where it raised or ran
// past the limit) and defined.json (whether the solution loaded and
// defines NAME as a function).
//
//     node harness.js <call limit seconds> <NAME> [stop]
//
// With `stop`, the first call that fails ends the run. As a check of one
// file, without running any call:
//
//     node harness.js check <file> <NAME>
//
// prints "compiles", "syntax error on line N: ...", "raised ..." or
// "does not define NAME".

const fs = require("fs");
const vm = require("vm");

const LOAD_LIMIT_MS = 2000;

function load(path, name) {
  // Run the file's top-level code; the state of its definition.
  const source = fs.readFileSync(path, "utf8");
  let script;
  try {
    script = new vm.Script(source, { filename: path });
  } catch (error) {
    const line = /:(\d+)$/.exec(String(error.stack).split("\n")[0]);
    const where = line ? ` on line ${line[1]}` : "";
    return `syntax error${where}: ${error.message}`;
  }
  try {
    script.runInThisContext({ timeout: LOAD_LIMIT_MS });
  } catch (error) {
    return `raised ${error && error.name ? error.name : "an error"}: ${
      error && error.message ? error.message : error
    }`;
  }
  const defined = vm.runInThisContext(`typeof ${name} === "function"`);
  return defined ? "compiles" : `does not define ${name}`;
}

function grade(limit, name, stop) {
  const calls = JSON.parse(fs.readFileSync("calls.json", "utf8"));
  const results = calls.map(() => null);
  const write = () => fs.writeFileSync("results.json", JSON.stringify(results));
  write();
  const state = load("solution.js", name);
  fs.writeFileSync("defined.json", JSON.stringify(state === "compiles"));
  if (state !== "compiles") {
    console.error(state);
    return;
  }
  const start = process.hrtime.bigint();
  for (let i = 0; i < calls.length; i++) {
    try {
      const value = vm.runInThisContext(calls[i], { timeout: limit * 1000 });
      const text = JSON.stringify(value === undefined ? null : value);
      results[i] = text === undefined ? null : text;
    } catch (error) {
      console.error(String(error && error.stack ? error.stack : error));
    }
    write();
    if (stop && results[i] === null) break;
  }
  const seconds = Number(process.hrtime.bigint() - start) / 1e9;
  fs.writeFileSync("seconds.json", JSON.stringify(seconds));
}

// The JSON of a function's result is never null where it returned a
// value: a real null (or undefined) result is the text "null", which the
// grader compares like any other; a JavaScript null in `results` means the
// call failed.
const args = process.argv.slice(process.argv[1] && process.argv[1].endsWith(".js") ? 2 : 1);
if (args[0] === "check") {
  console.log(load(args[1], args[2]));
} else {
  grade(Number(args[0]), args[1], args[2] === "stop");
}
