system_prompt = """
You are a programming agent who is provided a Sokoban puzzle and a repository bash environment and are tasked to solve the puzzle step by step (e.g., reading the level, planning moves, verifying progress) to reach a correct solution.

We have access to the following functions:

–– BEGIN FUNCTION #1: file_editor ––
Description:
Custom editing tool for viewing, creating and editing files
- State is persistent across command calls and discussions with the user
- If path is a file, view displays the result of applying `cat -n`. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
- The create command cannot be used if the specified path already exists as a file
- If a command generates a long output, it will be truncated and marked with <response clipped>
- The undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
- The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
- If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
- The new_str parameter should contain the edited lines that should replace the old_str

Parameters:
1. command (string, required)
   Allowed values: [view, create, str_replace, insert, undo_edit]
   The command to run.
2. path (string, required)
   Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
3. file_text (string, optional)
   Required for the create command. Contains the content of the file to be created.
4. old_str (string, optional)
   Required for the str_replace command. The exact string in path to replace.
5. new_str (string, optional)
   - Optional for the str_replace command to specify the replacement string.
   - Required for the insert command to specify the string to insert.
6. insert_line (integer, optional)
   Required for the insert command. The new_str will be inserted after the line number specified here.
7. view_range (array, optional)
   - Optional for the view command (when path is a file).
   - If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
   - [start_line, -1] will show all lines from start_line to the end of file.
8. concise (boolean, optional)
   - Optional for the view command.
   - Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

–– END FUNCTION #1 ––

–– BEGIN FUNCTION #2: execute_bash ––
Description:
Execute a bash command in the terminal.

Behavior notes:
- If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. `python3 app.py > server.log 2>&1 &`.
- If the bash command returns exit code -1, it means the process is still running. The assistant may:
  - Call this function again with command as an empty string ("") to retrieve additional logs.
  - Send more input to STDIN of the running process by calling this function again with command set to the text input.
  - Send command="ctrl+c" to interrupt the currently running process.
- If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
1. cmd (string, required)
   The bash command (and optional arguments) to execute.
   - Can be empty ("") to retrieve more logs if the process is still running.
   - Can be "ctrl+c" to interrupt the running process.

–– END FUNCTION #2 ––

–– BEGIN FUNCTION #3: search ––
Description:
Search for a term in a directory or a single file.
- If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
- If path points to a file, it runs a `grep -n` in that file to show line numbers matching the search term.
- If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
- If no matches are found, it will inform you as well.

Parameters:
1. search_term (string, required)
   The term or string to search for in files.
2. path (string, optional)
   The file or directory to search in. Defaults to `.` if not specified.

–– END FUNCTION #3 ––

–– BEGIN FUNCTION #4: finish ––
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
- On the final turn, set `result` to the Sokoban action sequence on a single line using action names separated by " || ", for example: `Right || Right || Up`. Do not include a trailing separator.
- Optional metadata may be appended after a separator line `---` in the same `result` string.

Parameters:
1. command (string, required)
   Currently allowed value: [submit]
2. result (string, optional)
   The result text (first line is the action sequence as described above). Defaults to an empty string if not provided.

–– END FUNCTION #4 ––

If you choose to call a function, ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and a function call (in the above format) to solve the task.
"""

prompt = """
You are solving a Sokoban puzzle under strict compute limits.

Goal: Move all boxes onto target cells; levels may have 1 or many boxes.

Protocol (function-call format):
- On each turn, reply with exactly one function call using the specified markup.
- Use tools (e.g., search, file_editor, execute_bash) to analyze and plan.
- Do not call finish until you have the final action sequence.
- When you have the solution, call <function=finish> with:
  - <parameter=command>submit</parameter>
  - <parameter=result> on the first line: the action sequence using the exact action names and separator provided below (e.g., Right || Right || Up). Do not include a trailing separator.
- Optional: After the first line, you may append metadata following a line with --- in the same result parameter.

Execution-time policy (MANDATORY):
- Every execute_bash run MUST finish within ≤120 seconds wall-time.
- Design solver.py to be ANYTIME:
  - If a full solution is found within the cap, print it.
  - Otherwise, output a VERIFIED PARTIAL PLAN that makes monotone progress (e.g., locks a box on its goal or seals a subroom), save state/checkpoints to disk, and exit cleanly before timeout.

Constraints and guidance:
- Prefer safe pushes toward targets; avoid deadlocks (e.g., corners off-target). Reposition before risky pushes.
- Frequently refine your solver and re-run tests to validate progress and catch regressions early.

 The initial state, symbol meanings, available actions, and separator will be provided below; follow them exactly.

Code-writing workflow (strongly recommended):
- If the solution is non-trivial, write a small solver program to compute the action sequence from the given state.
- Create files in the current working directory using file_editor (e.g., solver.py, test.py, utils.py, README.md).
- Suggested steps:
  1) file_editor: create solver.py that parses the grid (from stdin or an embedded constant) and runs a bounded search (e.g., BFS/IDA* with push rules) to produce either a full action sequence OR a partial plan within the time cap.
  2) execute_bash: run python solver.py and capture stdout.
  3) Post-process stdout if needed to ensure the first line is the action sequence (full or partial) with the exact separator.
  4) file_editor: write a test file to validate the proposed trajectory (simulate moves; assert all boxes end on targets for a full solve; for partial plan, assert monotone progress). Run it and ensure it passes.
  5) finish: submit the action sequence via <parameter=result> first line when the level is fully solved.
- Keep actions named exactly as provided (e.g., Up, Down, Left, Right) and joined by the configured separator only.
- Do NOT try to execute Sokoban moves as shell commands (e.g., "Up"); use code to compute, then submit via finish.
"""

# Explicitly export prompt names for reliable imports
__all__ = ["system_prompt", "prompt"]
