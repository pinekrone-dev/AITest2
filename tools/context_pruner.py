"""
Context Pruner - Send less, get the same answer.

Strips unnecessary content from code and text files before sending
to Claude. Removes comments, docstrings, blank lines, and
boilerplate so you send only what matters. Typically cuts token
count by 30-60% with zero loss of useful context.

Usage:
    from tools.context_pruner import ContextPruner

    pruner = ContextPruner()

    # Prune a file
    result = pruner.prune_file("app.py")
    print(result["pruned_text"])  # smaller, cheaper to send
    print(result["savings"])      # "42% reduction (1,200 tokens saved)"

    # Prune with specific strategies
    result = pruner.prune_file("app.py", strategies=["comments", "docstrings", "blanks"])
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


CHARS_PER_TOKEN = 3.75  # conservative average


class ContextPruner:
    """Strip unnecessary content from code before sending to Claude."""

    def prune_text(
        self,
        text: str,
        strategies: list[str] | None = None,
        is_code: bool = True,
    ) -> dict[str, Any]:
        """
        Prune text using specified strategies.

        Strategies:
            comments     - Remove single-line comments (# // --)
            docstrings   - Remove Python triple-quote docstrings
            blanks       - Collapse multiple blank lines to one
            imports      - Remove import statements (Python)
            type_hints   - Strip type annotations (Python)
            logging      - Remove logging/print statements
            boilerplate  - Remove common boilerplate (if __name__, etc.)
        """
        if strategies is None:
            strategies = ["comments", "docstrings", "blanks"]

        original_tokens = int(len(text) / CHARS_PER_TOKEN)
        result = text

        for strategy in strategies:
            if strategy == "comments":
                result = self._strip_comments(result)
            elif strategy == "docstrings":
                result = self._strip_docstrings(result)
            elif strategy == "blanks":
                result = self._collapse_blanks(result)
            elif strategy == "imports":
                result = self._strip_imports(result)
            elif strategy == "type_hints":
                result = self._strip_type_hints(result)
            elif strategy == "logging":
                result = self._strip_logging(result)
            elif strategy == "boilerplate":
                result = self._strip_boilerplate(result)

        pruned_tokens = int(len(result) / CHARS_PER_TOKEN)
        saved = original_tokens - pruned_tokens
        pct = (saved / original_tokens * 100) if original_tokens > 0 else 0

        return {
            "pruned_text": result,
            "original_tokens": original_tokens,
            "pruned_tokens": pruned_tokens,
            "tokens_saved": saved,
            "savings": f"{pct:.0f}% reduction ({saved:,} tokens saved)",
        }

    def prune_file(
        self,
        filepath: str,
        strategies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Prune a file and return the result."""
        path = Path(filepath)
        text = path.read_text(errors="replace")
        is_code = path.suffix.lower() in {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
            ".rs", ".cpp", ".c", ".rb", ".php",
        }
        result = self.prune_text(text, strategies, is_code)
        result["file"] = str(path)
        return result

    def prune_for_review(self, filepath: str) -> dict[str, Any]:
        """
        Aggressive pruning for code review context.
        Keeps function signatures and logic, strips everything else.
        """
        return self.prune_file(
            filepath,
            strategies=["comments", "docstrings", "blanks", "imports", "logging"],
        )

    def prune_for_bug_fix(self, filepath: str) -> dict[str, Any]:
        """
        Light pruning for bug fixing.
        Keeps comments (might be relevant) but strips blanks and docstrings.
        """
        return self.prune_file(
            filepath,
            strategies=["docstrings", "blanks"],
        )

    def extract_skeleton(self, filepath: str) -> dict[str, Any]:
        """
        Extract only function/class signatures from a Python file.
        Gives Claude the structure without the body. Massive token savings
        when you just need Claude to understand the API surface.
        """
        path = Path(filepath)
        text = path.read_text(errors="replace")
        lines = text.split("\n")
        skeleton_lines = []

        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("class ")
                or stripped.startswith("def ")
                or stripped.startswith("async def ")
                or stripped.startswith("@")
            ):
                skeleton_lines.append(line)
            elif stripped.startswith("self.") and "=" in stripped:
                # Instance variable assignments in __init__
                skeleton_lines.append(line)

        skeleton = "\n".join(skeleton_lines)
        original_tokens = int(len(text) / CHARS_PER_TOKEN)
        skeleton_tokens = int(len(skeleton) / CHARS_PER_TOKEN)
        saved = original_tokens - skeleton_tokens
        pct = (saved / original_tokens * 100) if original_tokens > 0 else 0

        return {
            "skeleton": skeleton,
            "original_tokens": original_tokens,
            "skeleton_tokens": skeleton_tokens,
            "tokens_saved": saved,
            "savings": f"{pct:.0f}% reduction ({saved:,} tokens saved)",
            "file": str(path),
        }

    # --- Internal strategies ---

    def _strip_comments(self, text: str) -> str:
        """Remove single-line comments."""
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            # Skip pure comment lines
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Remove inline comments (but not URLs with //)
            if "#" in line and not re.search(r'https?://', line):
                # Only strip if # isn't inside a string
                in_string = False
                quote_char = None
                idx = -1
                for i, ch in enumerate(line):
                    if ch in ('"', "'") and not in_string:
                        in_string = True
                        quote_char = ch
                    elif ch == quote_char and in_string:
                        in_string = False
                    elif ch == "#" and not in_string:
                        idx = i
                        break
                if idx > 0:
                    line = line[:idx].rstrip()
            result.append(line)
        return "\n".join(result)

    def _strip_docstrings(self, text: str) -> str:
        """Remove Python triple-quote docstrings."""
        # Match both ''' and \"\"\" docstrings
        text = re.sub(r'"""[\s\S]*?"""', '""""""', text)
        text = re.sub(r"'''[\s\S]*?'''", "''''''", text)
        return text

    def _collapse_blanks(self, text: str) -> str:
        """Collapse multiple blank lines to single blank line."""
        return re.sub(r"\n{3,}", "\n\n", text)

    def _strip_imports(self, text: str) -> str:
        """Remove import lines."""
        lines = text.split("\n")
        result = []
        skip_multiline = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                if line.rstrip().endswith("\\") or "(" in line and ")" not in line:
                    skip_multiline = True
                continue
            if skip_multiline:
                if ")" in line or not line.strip().endswith("\\"):
                    skip_multiline = False
                continue
            result.append(line)
        return "\n".join(result)

    def _strip_type_hints(self, text: str) -> str:
        """Remove type annotations from Python code."""
        # Remove function return types
        text = re.sub(r"\)\s*->\s*[^:]+:", "):", text)
        # Remove parameter type hints
        text = re.sub(r":\s*(?:str|int|float|bool|list|dict|tuple|set|Any|Optional)\[?[^\],=)]*\]?", "", text)
        return text

    def _strip_logging(self, text: str) -> str:
        """Remove logging and print statements."""
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(p) for p in [
                "print(", "logger.", "logging.", "log.",
                "console.log(", "console.error(", "console.warn(",
            ]):
                continue
            result.append(line)
        return "\n".join(result)

    def _strip_boilerplate(self, text: str) -> str:
        """Remove common boilerplate patterns."""
        lines = text.split("\n")
        result = []
        skip_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('if __name__ == "__main__"') or stripped.startswith("if __name__ == '__main__'"):
                skip_block = True
                continue
            if skip_block:
                if line and not line[0].isspace():
                    skip_block = False
                else:
                    continue
            result.append(line)
        return "\n".join(result)


def main():
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m tools.context_pruner path/to/file.py")
        print("  python -m tools.context_pruner path/to/file.py --skeleton")
        print("  python -m tools.context_pruner path/to/file.py --aggressive")
        sys.exit(1)

    pruner = ContextPruner()
    filepath = sys.argv[1]

    if "--skeleton" in sys.argv:
        result = pruner.extract_skeleton(filepath)
        print(result["skeleton"])
        print(f"\n--- {result['savings']} ---")
    elif "--aggressive" in sys.argv:
        result = pruner.prune_for_review(filepath)
        print(result["pruned_text"])
        print(f"\n--- {result['savings']} ---")
    else:
        result = pruner.prune_file(filepath)
        print(result["pruned_text"])
        print(f"\n--- {result['savings']} ---")


if __name__ == "__main__":
    main()
