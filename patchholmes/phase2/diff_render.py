"""Render raw `git diff` text into an LLM-agent-readable form.

Pure rules, no LLM, no SDK dependency. Three stages:

    raw diff text
        │
        ▼
    [1] parse_diff       → list[FileChange]
        │
        ▼
    [2] classify_files   → tag + priority per file
        │
        ▼
    [3] render_for_agent → final agent-readable string under char budget

See `docs/diff-render-for-agent.md` for the design rationale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


FileTag = Literal["source", "test", "doc", "config", "fixture", "other"]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class FileChange:
    path: str
    raw_block: str
    is_binary: bool = False
    adds: int = 0
    dels: int = 0
    hunk_count: int = 0
    tag: FileTag = "other"
    priority: int = 0

    @property
    def total_changed(self) -> int:
        return self.adds + self.dels


# ---------------------------------------------------------------------------
# Stage 1: parse_diff
# ---------------------------------------------------------------------------

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")


def _path_from_diff_git_line(line: str) -> str:
    """Extract the post-image path from a `diff --git a/X b/Y` line.

    Prefer the `b/` path because that's the file after the change (relevant for
    renames/new files). Falls back to whitespace splitting if regex misses.
    """
    m = _DIFF_GIT_RE.match(line.rstrip("\n"))
    if m:
        return m.group(2).strip()
    parts = line.split()
    if len(parts) >= 4:
        p = parts[3]
        return p[2:] if p.startswith("b/") else p
    return ""


def parse_diff(text: str) -> list[FileChange]:
    """Split a raw multi-file diff into one FileChange per file.

    Recognised line prefixes:
        diff --git a/X b/Y     → start of a new file block
        Binary files ... differ → marks block as binary
        @@ -X,Y +X,Y @@        → hunk header (counts hunks)
        +xxx (not +++)         → added line (counts as add)
        -xxx (not ---)         → deleted line (counts as del)
    """
    if not text:
        return []

    files: list[FileChange] = []
    current: FileChange | None = None
    current_lines: list[str] = []

    def _commit_current():
        nonlocal current, current_lines
        if current is not None:
            current.raw_block = "\n".join(current_lines)
            files.append(current)
        current = None
        current_lines = []

    for line in text.splitlines():
        if line.startswith("diff --git "):
            _commit_current()
            path = _path_from_diff_git_line(line)
            current = FileChange(path=path, raw_block="")
            current_lines = [line]
            continue

        if current is None:
            # diff text without a header — skip
            continue

        current_lines.append(line)

        if line.startswith("Binary files") and "differ" in line:
            current.is_binary = True
        elif line.startswith("@@"):
            current.hunk_count += 1
        elif line.startswith("+") and not line.startswith("+++"):
            current.adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            current.dels += 1

    _commit_current()
    return files


# ---------------------------------------------------------------------------
# Stage 2: classify_files
# ---------------------------------------------------------------------------

SOURCE_EXTS = {
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx",
    ".py", ".pyx", ".pyi",
    ".java", ".kt", ".scala",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs",
    ".rb", ".php", ".cs",
    ".m", ".mm", ".swift",
    ".lua", ".pl", ".sh", ".bash", ".zsh",
    ".sql",
}
DOC_EXTS = {".md", ".rst", ".txt", ".adoc"}
CONFIG_NAMES = {
    "setup.py", "setup.cfg", "pyproject.toml",
    "Makefile", "makefile", "CMakeLists.txt", "configure", "configure.ac",
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Gemfile", "Gemfile.lock", "Cargo.toml", "Cargo.lock",
    "go.mod", "go.sum",
    "Dockerfile", ".dockerignore",
}
CONFIG_EXTS = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".lock"}

BASE_PRIORITY: dict[FileTag, int] = {
    "source":  100,
    "test":     40,
    "config":   30,
    "doc":      10,
    "fixture":   0,
    "other":    20,
}

# Path-component stopwords we don't count as "matching CVE description".
_PATH_STOPWORDS = {
    "py", "src", "tests", "test", "lib", "libs", "include", "headers",
    "main", "java", "com", "net", "org", "io", "core", "common", "util", "utils",
    "modules", "module", "pkg", "package", "build", "dist",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _ext(path: str) -> str:
    if "." not in path.rsplit("/", 1)[-1]:
        return ""
    return "." + path.rsplit(".", 1)[-1].lower()


def _filename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _classify_one(fc: FileChange) -> FileTag:
    if fc.is_binary:
        return "fixture"

    p = fc.path.lower()
    fname = _filename(p)

    # tests have precedence over generic source
    if "/tests/" in p or "/test/" in p or "/testing/" in p:
        return "test"
    if p.startswith(("tests/", "test/", "testing/")):
        return "test"
    if fname.startswith(("test_", "tests_")):
        return "test"
    if fname.endswith(("_test.py", "_test.go", "_test.java", "_tests.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts")):
        return "test"

    # docs
    if "/docs/" in p or "/doc/" in p:
        return "doc"
    if p.startswith(("docs/", "doc/", "documentation/")):
        return "doc"
    if _ext(p) in DOC_EXTS:
        return "doc"
    if fname in {"readme", "changelog", "changes", "notice", "license", "authors", "contributors"}:
        return "doc"

    # config
    if _filename(fc.path) in CONFIG_NAMES:
        return "config"
    if _ext(p) in CONFIG_EXTS:
        return "config"

    # source
    if _ext(p) in SOURCE_EXTS:
        return "source"

    return "other"


def _compute_priority(fc: FileChange, cve_tokens: set[str]) -> int:
    p = BASE_PRIORITY[fc.tag]

    # reward size up to a cap
    p += min(20, fc.total_changed)

    # penalise huge changes (refactors, vendored bumps)
    if fc.total_changed > 500:
        p -= 10

    # token overlap between file path and CVE description
    path_tokens = {t.lower() for t in _TOKEN_RE.findall(fc.path)}
    path_tokens -= _PATH_STOPWORDS
    if path_tokens & cve_tokens:
        p += 15

    return p


def classify_files(files: list[FileChange], cve_desc: str) -> list[FileChange]:
    """Set `tag` and `priority` on each FileChange in place. Returns the list."""
    cve_tokens = {t.lower() for t in _TOKEN_RE.findall(cve_desc or "")}
    cve_tokens = {t for t in cve_tokens if len(t) >= 3}
    for fc in files:
        fc.tag = _classify_one(fc)
        fc.priority = _compute_priority(fc, cve_tokens)
    return files


# ---------------------------------------------------------------------------
# Compression: drop context lines, keep +/- and headers
# ---------------------------------------------------------------------------

def compress_block(block: str, max_chars: int) -> str:
    """Compress a single-file diff block by dropping context lines.

    Always kept:
        - diff --git, index, ---/+++, mode lines, rename/copy lines
        - @@ hunk headers
        - + and - change lines
    Dropped:
        - unchanged context lines (start with a single space)

    If the result still exceeds `max_chars`, head-and-tail truncate with a marker.
    """
    keep: list[str] = []
    for line in block.splitlines():
        if (
            line.startswith("diff --git")
            or line.startswith("index ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("@@")
            or line.startswith("new file mode")
            or line.startswith("deleted file mode")
            or line.startswith("old mode")
            or line.startswith("new mode")
            or line.startswith("similarity index")
            or line.startswith("dissimilarity index")
            or line.startswith("rename from")
            or line.startswith("rename to")
            or line.startswith("copy from")
            or line.startswith("copy to")
            or line.startswith("Binary files")
            or (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        ):
            keep.append(line)
        # else: a context line (starts with " ") or blank — drop

    compressed = "\n".join(keep)
    if len(compressed) <= max_chars:
        return compressed

    half = max(200, max_chars // 2 - 30)
    return compressed[:half] + "\n... [middle truncated] ...\n" + compressed[-half:]


# ---------------------------------------------------------------------------
# Stage 3: render_for_agent
# ---------------------------------------------------------------------------

MANIFEST_FILE_LIMIT = 50  # Cap how many file lines we list per commit


def _format_file_manifest(files: list[FileChange]) -> str:
    """Render the file manifest. Capped at MANIFEST_FILE_LIMIT lines.

    Always shows:
      - total file count + per-tag breakdown
      - top-MANIFEST_FILE_LIMIT files by priority (input is already sorted)
      - a "more files not shown" footer with guidance when capped

    Real security-fix commits touch < 10 files (Pillow p90 = 8), so the cap
    only triggers on big-refactor / repo-init / vendored-bump commits, which
    are almost never the answer anyway.
    """
    total = len(files)

    # Tag breakdown for the always-visible summary line
    tag_counts: dict[str, int] = {}
    for fc in files:
        tag_counts[fc.tag] = tag_counts.get(fc.tag, 0) + 1
    breakdown = ", ".join(
        f"{tag_counts.get(t, 0)} {t}"
        for t in ("source", "test", "config", "doc", "fixture", "other")
        if tag_counts.get(t, 0) > 0
    )

    if total <= MANIFEST_FILE_LIMIT:
        header = f"[Files changed: {total} — {breakdown}]"
        shown = files
        hidden = 0
    else:
        header = (
            f"[Files changed: {total} — {breakdown}; "
            f"showing top-{MANIFEST_FILE_LIMIT} by relevance]"
        )
        shown = files[:MANIFEST_FILE_LIMIT]
        hidden = total - MANIFEST_FILE_LIMIT

    lines = [header]
    for fc in shown:
        marker = "(binary)" if fc.is_binary else f"+{fc.adds}/-{fc.dels}"
        lines.append(f"  {fc.path:<60}  {fc.tag:<8}  {marker}")

    if hidden > 0:
        lines.append(
            f"  [+ {hidden} more files not shown — typically low-priority "
            f"(binaries, docs, etc.). Large commits with 50+ files are "
            f"usually refactors or repo init, rarely security fixes. "
            f"Call read_file_diff(commit_id, path) if you suspect a specific "
            f"file you remember from elsewhere.]"
        )

    return "\n".join(lines)


def render_for_agent(
    commit_msg: str,
    files: list[FileChange],
    char_budget: int = 8000,
) -> str:
    """Render a commit (already-classified files) into an agent-readable string.

    Layout:
        MSG: <commit message>

        [Files changed: N]
          path1   tag   +adds/-dels
          ...

        [Diff content, budget ~8000 chars]
        --- path1 ---
        <diff block or compressed block>
        --- path2 ---
        ...

        [Binary files skipped: K]
        [Files not shown (use read_file_diff): pathA, pathB, ...]
    """
    sorted_files = sorted(files, key=lambda f: -f.priority)

    out: list[str] = []
    msg = (commit_msg or "").strip()
    out.append(f"MSG: {msg}" if msg else "MSG: (empty)")
    out.append("")
    out.append(_format_file_manifest(sorted_files))
    out.append("")
    out.append(f"[Diff content, budget {char_budget} chars]")

    remaining = char_budget
    binary_skipped = 0
    truncated_paths: list[str] = []

    for fc in sorted_files:
        if fc.is_binary:
            binary_skipped += 1
            continue
        if remaining <= 200:
            truncated_paths.append(fc.path)
            continue

        block = fc.raw_block
        header = f"--- {fc.path} ---"
        block_with_header = header + "\n" + block
        if len(block_with_header) <= remaining:
            out.append(block_with_header)
            remaining -= len(block_with_header) + 1  # +1 for the join newline
            continue

        # need compression for this file
        budget_for_block = max(400, remaining - len(header) - 1)
        compressed = compress_block(block, budget_for_block)
        out.append(header)
        out.append(compressed)
        out.append(
            f"[... {fc.path}: full diff {len(block)} chars, "
            f"call read_file_diff to see complete content]"
        )
        truncated_paths.append(fc.path)
        remaining = 0

    if binary_skipped:
        out.append("")
        out.append(f"[Binary files skipped: {binary_skipped} (no content shown)]")
    if truncated_paths:
        # only list files truncated AFTER binary skipping
        shown_list = ", ".join(truncated_paths[:5])
        extra = f" (+{len(truncated_paths)-5} more)" if len(truncated_paths) > 5 else ""
        out.append(f"[Files truncated — call read_file_diff to inspect: {shown_list}{extra}]")

    return "\n".join(out)


def render_single_file(fc: FileChange, char_budget: int = 16000) -> str:
    """Render one file's diff for the read_file_diff tool. Larger budget than render_for_agent."""
    if fc.is_binary:
        return f"--- {fc.path} ---\n[Binary file, no diff content. adds={fc.adds} dels={fc.dels}]"

    header = f"--- {fc.path} ---"
    block = fc.raw_block
    if len(block) <= char_budget - len(header) - 1:
        return header + "\n" + block
    compressed = compress_block(block, char_budget - len(header) - 1)
    return (
        header + "\n" + compressed
        + f"\n[... {fc.path}: original {len(block)} chars, compressed to fit budget]"
    )
