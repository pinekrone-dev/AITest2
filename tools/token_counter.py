"""
Token Counter - Know what it costs before you send it.

Estimates token count and dollar cost for any text, file, or prompt
before you send it to the Claude API. Prevents surprise bills and
helps you decide whether to prune content first.

Usage:
    from tools.token_counter import TokenCounter

    tc = TokenCounter()
    tc.estimate("Write me a marketing email for brokers")
    tc.estimate_file("app.py")
    tc.estimate_conversation(messages)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


# Approximate token ratios (conservative estimates based on Claude tokenizer behavior)
# 1 token ~ 4 characters for English text
# 1 token ~ 3.5 characters for code
# These are intentionally conservative so you never undershoot cost
CHARS_PER_TOKEN_TEXT = 4.0
CHARS_PER_TOKEN_CODE = 3.5

# Claude API pricing per million tokens (as of early 2025, update as needed)
PRICING = {
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # Cached input pricing (prompt caching)
    "claude-opus-4-6-cached": {"input": 1.50, "output": 75.00},
    "claude-sonnet-4-6-cached": {"input": 0.30, "output": 15.00},
    "claude-haiku-4-5-20251001-cached": {"input": 0.08, "output": 4.00},
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt", ".cs",
    ".html", ".css", ".scss", ".sql", ".sh", ".bash", ".yml",
    ".yaml", ".json", ".toml", ".xml",
}


class TokenCounter:
    """Estimate tokens and costs before sending to Claude."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        if model not in PRICING:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(PRICING.keys())}"
            )

    def count_tokens(self, text: str, is_code: bool = False) -> int:
        """Estimate token count for a string."""
        if not text:
            return 0
        ratio = CHARS_PER_TOKEN_CODE if is_code else CHARS_PER_TOKEN_TEXT
        return max(1, int(len(text) / ratio))

    def estimate(
        self,
        text: str,
        is_code: bool = False,
        expected_output_tokens: int = 500,
    ) -> dict[str, Any]:
        """Estimate tokens and cost for a prompt."""
        input_tokens = self.count_tokens(text, is_code)
        prices = PRICING[self.model]
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (expected_output_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost

        result = {
            "input_tokens": input_tokens,
            "expected_output_tokens": expected_output_tokens,
            "total_tokens": input_tokens + expected_output_tokens,
            "model": self.model,
            "input_cost": f"${input_cost:.6f}",
            "output_cost": f"${output_cost:.6f}",
            "total_cost": f"${total_cost:.6f}",
            "tip": None,
        }

        # Give actionable advice
        if input_tokens > 10_000:
            result["tip"] = (
                f"Large input ({input_tokens:,} tokens). "
                "Consider using context_pruner to strip comments/whitespace, "
                "or file_differ to send only changed sections."
            )
        elif input_tokens > 50_000:
            result["tip"] = (
                f"Very large input ({input_tokens:,} tokens). "
                "Strongly recommend pruning or splitting this into smaller chunks."
            )

        return result

    def estimate_file(
        self, filepath: str, expected_output_tokens: int = 500
    ) -> dict[str, Any]:
        """Estimate tokens and cost for sending a file to Claude."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = path.read_text(errors="replace")
        is_code = path.suffix.lower() in CODE_EXTENSIONS
        result = self.estimate(text, is_code, expected_output_tokens)
        result["file"] = str(path)
        result["file_size_bytes"] = path.stat().st_size
        result["lines"] = text.count("\n") + 1
        return result

    def estimate_directory(
        self,
        dirpath: str,
        extensions: set[str] | None = None,
        expected_output_tokens: int = 1000,
    ) -> dict[str, Any]:
        """Estimate cost of sending an entire directory to Claude."""
        root = Path(dirpath)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {dirpath}")

        total_tokens = 0
        file_count = 0
        largest_files: list[tuple[int, str]] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if extensions and path.suffix.lower() not in extensions:
                continue
            # Skip binary / lock files
            if path.suffix in {".lock", ".pyc", ".pyo", ".whl", ".egg"}:
                continue

            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue

            is_code = path.suffix.lower() in CODE_EXTENSIONS
            tokens = self.count_tokens(text, is_code)
            total_tokens += tokens
            file_count += 1
            largest_files.append((tokens, str(path.relative_to(root))))

        largest_files.sort(reverse=True)
        prices = PRICING[self.model]
        input_cost = (total_tokens / 1_000_000) * prices["input"]
        output_cost = (expected_output_tokens / 1_000_000) * prices["output"]

        return {
            "directory": str(root),
            "files_scanned": file_count,
            "total_input_tokens": total_tokens,
            "expected_output_tokens": expected_output_tokens,
            "input_cost": f"${input_cost:.6f}",
            "output_cost": f"${output_cost:.6f}",
            "total_cost": f"${input_cost + output_cost:.6f}",
            "top_5_largest_files": [
                {"file": f, "tokens": t} for t, f in largest_files[:5]
            ],
        }

    def compare_models(
        self, text: str, is_code: bool = False, expected_output_tokens: int = 500
    ) -> list[dict[str, Any]]:
        """Compare cost across all models so you pick the cheapest that works."""
        input_tokens = self.count_tokens(text, is_code)
        results = []
        for model, prices in PRICING.items():
            if model.endswith("-cached"):
                continue
            input_cost = (input_tokens / 1_000_000) * prices["input"]
            output_cost = (expected_output_tokens / 1_000_000) * prices["output"]
            results.append({
                "model": model,
                "input_tokens": input_tokens,
                "total_cost": f"${input_cost + output_cost:.6f}",
                "input_cost": f"${input_cost:.6f}",
                "output_cost": f"${output_cost:.6f}",
            })
        results.sort(key=lambda r: float(r["total_cost"].replace("$", "")))
        return results

    def savings_with_cache(
        self, text: str, is_code: bool = False, expected_output_tokens: int = 500
    ) -> dict[str, Any]:
        """Show how much you save using prompt caching."""
        input_tokens = self.count_tokens(text, is_code)
        cached_model = f"{self.model}-cached"
        if cached_model not in PRICING:
            return {"error": f"No cached pricing for {self.model}"}

        normal = PRICING[self.model]
        cached = PRICING[cached_model]

        normal_cost = (input_tokens / 1_000_000) * normal["input"]
        cached_cost = (input_tokens / 1_000_000) * cached["input"]
        savings = normal_cost - cached_cost

        return {
            "input_tokens": input_tokens,
            "normal_input_cost": f"${normal_cost:.6f}",
            "cached_input_cost": f"${cached_cost:.6f}",
            "savings_per_call": f"${savings:.6f}",
            "savings_pct": f"{(savings / normal_cost * 100) if normal_cost > 0 else 0:.1f}%",
            "tip": "Use prompt caching for system prompts, templates, and context that repeats across calls.",
        }


def main():
    """CLI entry point for quick estimates."""
    import sys

    tc = TokenCounter()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m tools.token_counter 'your prompt text here'")
        print("  python -m tools.token_counter --file path/to/file.py")
        print("  python -m tools.token_counter --dir path/to/directory")
        print("  python -m tools.token_counter --compare 'your prompt'")
        sys.exit(1)

    if sys.argv[1] == "--file":
        result = tc.estimate_file(sys.argv[2])
    elif sys.argv[1] == "--dir":
        result = tc.estimate_directory(sys.argv[2])
    elif sys.argv[1] == "--compare":
        result = tc.compare_models(" ".join(sys.argv[2:]))
    else:
        result = tc.estimate(" ".join(sys.argv[1:]))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
