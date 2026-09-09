"""Launcher: run an external Favia checkout's main_local.py with a scroll cap
that ACTUALLY breaks runaway `for _ in range(N): scroll_file(...)` loops.

Why: upstream ScrollFileTool.forward increments a counter and, past MAX_SCROLLS,
merely RETURNS an error string. A literal Python loop the agent writes in one
code step ignores the return value, so forward() still fires hundreds of
thousands of times — each creating an OpenTelemetry span (2GB+ trace) and
holding the GIL — which starves the other ThreadPoolExecutor workers and
collapses throughput to ~1 effective worker (observed: one agent invoked
scroll_file 263,597 times; several such agents dominated 97-98% of calls).

Fix: past the cap, forward() RAISES. The exception unwinds out of the agent's
code block, aborting the loop on the spot; smolagents 1.21.x catches it as a
tool error and ends the ReAct step. Worst case per pair drops from ~260k scroll
calls to MAX_SCROLLS * max_steps (~750). Normal agents (median 5-6 scrolls)
never reach the cap, so their behaviour is byte-for-byte unchanged.

Point this at your own Favia checkout via the FAVIA_EVAL_DIR environment
variable (the directory that contains main_local.py and the sibling src/).
All argv are forwarded verbatim to main_local.main().

Requires: an external Favia checkout + smolagents.
"""
import os
import sys

# Directory of the external Favia checkout that holds main_local.py (and src/).
EVAL_DIR = os.environ.get("FAVIA_EVAL_DIR", "./external/favia/evaluation")
EVAL_DIR = os.path.abspath(EVAL_DIR)
FAVIA_DIR = os.path.dirname(EVAL_DIR)  # contains src/
os.chdir(EVAL_DIR)                      # main_local does load_dotenv(".env") + relative imports
sys.path.insert(0, FAVIA_DIR)          # for `import src.*`
sys.path.insert(0, EVAL_DIR)           # for `import main_local`

# Patch the class BEFORE main_local imports PatchClassifier -> ScrollFileTool.
from src.tools import scroll_file_tool as _sft


def _forward_raise(self, direction: str):
    self._calls += 1
    if self._calls > _sft.MAX_SCROLLS:
        raise RuntimeError(
            f"scroll_file hard limit ({_sft.MAX_SCROLLS}) reached for this task — "
            f"stop scrolling; use code_search / file_search / open_file, or submit "
            f"your final answer now."
        )
    return self.windowed_file.scroll(direction)


_sft.ScrollFileTool.forward = _forward_raise
print(f"[capped-launcher] ScrollFileTool.forward RAISES past MAX_SCROLLS={_sft.MAX_SCROLLS}",
      flush=True)

import main_local  # noqa: E402  (import after patch on purpose)

if __name__ == "__main__":
    main_local.main()
