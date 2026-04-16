"""
File Differ - Send changes, not entire files.

When you need Claude to review or work on code you've modified,
send only the diff instead of the full file. A 1000-line file
with 5 changed lines becomes a 20-token payload instead of 3000.

Also supports "smart context" - sends the diff plus just enough
surrounding code for Claude to understand what changed.

Usage:
    from tools.file_differ import FileDiffer

    fd = FileDiffer()

    # Compare current file to last version sent to Claude
    result = fd.diff_file("app.py")
    print(result["diff_prompt"])  # send this to Claude instead of the full file

    # Compare two versions
    result = fd.diff_texts(old_code, new_code)

    # Smart context: diff + surrounding functions
    result = fd.smart_diff("app.py", context_lines=10)
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SNAPSHOT_DIR = ".file_snapshots"
CHARS_PER_TOKEN = 3.75


class FileDiffer:
    """Send diffs instead of full files to save tokens."""

    def __init__(self, snapshot_dir: str | None = None):
        self.snapshot_dir = Path(snapshot_dir or SNAPSHOT_DIR)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, filepath: str) -> Path:
        """Get the snapshot storage path for a file."""
        key = hashlib.sha256(filepath.encode()).hexdigest()[:12]
        return self.snapshot_dir / f"{key}.txt"

    def snapshot(self, filepath: str):
        """Take a snapshot of a file (save its current state)."""
        path = Path(filepath)
        text = path.read_text(errors="replace")
        snap = self._snapshot_path(str(path.resolve()))
        snap.write_text(text)
        return {
            "file": str(path),
            "lines": text.count("\n") + 1,
            "snapshot_saved": True,
        }

    def snapshot_directory(self, dirpath: str, extensions: set[str] | None = None):
        """Snapshot all files in a directory."""
        root = Path(dirpath)
        count = 0
        for path in root.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            if extensions and path.suffix.lower() not in extensions:
                continue
            self.snapshot(str(path))
            count += 1
        return {"directory": str(root), "files_snapshotted": count}

    def diff_file(
        self, filepath: str, context_lines: int = 3
    ) -> dict[str, Any]:
        """
        Diff current file against its last snapshot.
        Returns a compact prompt with just the changes.
        """
        path = Path(filepath)
        current = path.read_text(errors="replace")
        snap_path = self._snapshot_path(str(path.resolve()))

        if not snap_path.exists():
            # No snapshot exists - take one and return full file info
            self.snapshot(filepath)
            return {
                "diff_prompt": f"[First time seeing {path.name} - full file sent]",
                "full_file_needed": True,
                "tokens_current": int(len(current) / CHARS_PER_TOKEN),
                "tip": "Snapshot saved. Next time only the diff will be sent.",
            }

        previous = snap_path.read_text()

        if current == previous:
            return {
                "diff_prompt": f"[No changes to {path.name}]",
                "changed": False,
                "tokens_saved": int(len(current) / CHARS_PER_TOKEN),
            }

        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            previous.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"{path.name} (previous)",
            tofile=f"{path.name} (current)",
            n=context_lines,
        ))

        diff_text = "".join(diff_lines)

        # Build a prompt-friendly version
        diff_prompt = (
            f"Here are the changes to {path.name}:\n\n"
            f"```diff\n{diff_text}```\n"
        )

        full_tokens = int(len(current) / CHARS_PER_TOKEN)
        diff_tokens = int(len(diff_prompt) / CHARS_PER_TOKEN)
        saved = full_tokens - diff_tokens
        pct = (saved / full_tokens * 100) if full_tokens > 0 else 0

        # Update snapshot to current
        snap_path.write_text(current)

        return {
            "diff_prompt": diff_prompt,
            "changed": True,
            "lines_changed": len([l for l in diff_lines if l.startswith("+") or l.startswith("-")]),
            "full_file_tokens": full_tokens,
            "diff_tokens": diff_tokens,
            "tokens_saved": saved,
            "savings": f"{pct:.0f}% reduction ({saved:,} tokens saved)",
        }

    def diff_texts(
        self,
        old_text: str,
        new_text: str,
        label: str = "file",
        context_lines: int = 3,
    ) -> dict[str, Any]:
        """Diff two text strings directly."""
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"{label} (old)",
            tofile=f"{label} (new)",
            n=context_lines,
        ))

        diff_text = "".join(diff_lines)
        diff_prompt = f"Changes to {label}:\n\n```diff\n{diff_text}```\n"

        full_tokens = int(len(new_text) / CHARS_PER_TOKEN)
        diff_tokens = int(len(diff_prompt) / CHARS_PER_TOKEN)

        return {
            "diff_prompt": diff_prompt,
            "full_file_tokens": full_tokens,
            "diff_tokens": diff_tokens,
            "tokens_saved": full_tokens - diff_tokens,
        }

    def smart_diff(
        self, filepath: str, context_lines: int = 10
    ) -> dict[str, Any]:
        """
        Generate a diff with smart context - includes the surrounding
        function/class definitions so Claude understands the scope
        of changes without seeing the whole file.
        """
        path = Path(filepath)
        current = path.read_text(errors="replace")
        snap_path = self._snapshot_path(str(path.resolve()))

        if not snap_path.exists():
            self.snapshot(filepath)
            return self._extract_relevant_sections(current, path.name)

        previous = snap_path.read_text()

        if current == previous:
            return {
                "smart_prompt": f"[No changes to {path.name}]",
                "changed": False,
            }

        # Find changed line numbers
        current_lines = current.splitlines()
        previous_lines = previous.splitlines()
        matcher = difflib.SequenceMatcher(None, previous_lines, current_lines)

        changed_line_numbers = set()
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                for n in range(j1, j2):
                    changed_line_numbers.add(n)

        if not changed_line_numbers:
            snap_path.write_text(current)
            return {"smart_prompt": f"[No meaningful changes to {path.name}]", "changed": False}

        # Expand context around changes to include full function/class blocks
        sections = self._get_context_sections(current_lines, changed_line_numbers, context_lines)

        # Build the smart context prompt
        parts = [f"Changes to {path.name} with surrounding context:\n"]
        for start, end, lines in sections:
            parts.append(f"Lines {start + 1}-{end + 1}:")
            for i, line in enumerate(lines, start):
                marker = ">>>" if i in changed_line_numbers else "   "
                parts.append(f"{marker} {i + 1:4d} | {line}")
            parts.append("")

        smart_prompt = "\n".join(parts)
        full_tokens = int(len(current) / CHARS_PER_TOKEN)
        smart_tokens = int(len(smart_prompt) / CHARS_PER_TOKEN)

        snap_path.write_text(current)

        return {
            "smart_prompt": smart_prompt,
            "changed": True,
            "full_file_tokens": full_tokens,
            "smart_tokens": smart_tokens,
            "tokens_saved": full_tokens - smart_tokens,
            "sections_extracted": len(sections),
        }

    def _get_context_sections(
        self,
        lines: list[str],
        changed_lines: set[int],
        context: int,
    ) -> list[tuple[int, int, list[str]]]:
        """Extract sections around changed lines with enough context."""
        if not changed_lines:
            return []

        # Group changed lines into ranges
        sorted_changes = sorted(changed_lines)
        ranges = []
        start = sorted_changes[0]
        end = sorted_changes[0]

        for n in sorted_changes[1:]:
            if n <= end + context * 2:
                end = n
            else:
                ranges.append((start, end))
                start = n
                end = n
        ranges.append((start, end))

        # Expand each range with context and try to capture function boundaries
        sections = []
        for start, end in ranges:
            ctx_start = max(0, start - context)
            ctx_end = min(len(lines) - 1, end + context)

            # Try to expand to function/class boundaries
            for i in range(ctx_start, -1, -1):
                stripped = lines[i].strip()
                if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("async def "):
                    ctx_start = i
                    break

            section_lines = lines[ctx_start:ctx_end + 1]
            sections.append((ctx_start, ctx_end, section_lines))

        return sections

    def _extract_relevant_sections(
        self, text: str, filename: str
    ) -> dict[str, Any]:
        """For first-time files, extract just the structure + key logic."""
        lines = text.splitlines()
        relevant = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or stripped.startswith("async def ")
                or stripped.startswith("@")
                or "TODO" in stripped
                or "FIXME" in stripped
                or "BUG" in stripped
            ):
                # Include this line + a few after it
                for j in range(i, min(i + 3, len(lines))):
                    relevant.append(f"{j + 1:4d} | {lines[j]}")
                relevant.append("     ...")

        skeleton = "\n".join(relevant)
        full_tokens = int(len(text) / CHARS_PER_TOKEN)
        skeleton_tokens = int(len(skeleton) / CHARS_PER_TOKEN)

        return {
            "smart_prompt": f"Structure of {filename} (first time):\n\n{skeleton}",
            "full_file_needed": True,
            "full_file_tokens": full_tokens,
            "skeleton_tokens": skeleton_tokens,
            "tokens_saved": full_tokens - skeleton_tokens,
        }

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all saved snapshots."""
        snapshots = []
        for f in self.snapshot_dir.glob("*.txt"):
            snapshots.append({
                "key": f.stem,
                "size_bytes": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
        return snapshots

    def clear_snapshots(self) -> int:
        """Clear all snapshots."""
        count = 0
        for f in self.snapshot_dir.glob("*.txt"):
            f.unlink()
            count += 1
        return count


def main():
    import sys

    fd = FileDiffer()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m tools.file_differ path/to/file.py          # diff against snapshot")
        print("  python -m tools.file_differ path/to/file.py --smart   # smart context diff")
        print("  python -m tools.file_differ path/to/file.py --snap    # take snapshot only")
        print("  python -m tools.file_differ --list                     # list snapshots")
        print("  python -m tools.file_differ --clear                    # clear snapshots")
        sys.exit(1)

    if sys.argv[1] == "--list":
        for snap in fd.list_snapshots():
            print(f"  {snap['key']}  {snap['size_bytes']} bytes")
    elif sys.argv[1] == "--clear":
        count = fd.clear_snapshots()
        print(f"Cleared {count} snapshots.")
    elif "--snap" in sys.argv:
        result = fd.snapshot(sys.argv[1])
        print(f"Snapshot saved for {result['file']} ({result['lines']} lines)")
    elif "--smart" in sys.argv:
        result = fd.smart_diff(sys.argv[1])
        print(result["smart_prompt"])
        if "tokens_saved" in result:
            print(f"\n--- Saved {result['tokens_saved']} tokens ---")
    else:
        result = fd.diff_file(sys.argv[1])
        print(result["diff_prompt"])
        if "tokens_saved" in result:
            print(f"\n--- Saved {result['tokens_saved']} tokens ---")


if __name__ == "__main__":
    main()
