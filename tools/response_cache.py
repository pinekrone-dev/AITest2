"""
Response Cache - Never pay twice for the same answer.

Caches Claude API responses locally so identical or similar queries
return instantly without hitting the API. Tracks hit rates so you
can see exactly how much money you're saving.

Usage:
    from tools.response_cache import ResponseCache

    cache = ResponseCache()

    # Check cache before calling API
    cached = cache.get("Write a cold email for Vegas brokers")
    if cached:
        print(cached["response"])  # free
    else:
        response = call_claude_api(prompt)
        cache.set("Write a cold email for Vegas brokers", response)

    # See savings
    cache.stats()
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = ".claude_cache"
DEFAULT_TTL_HOURS = 72  # Cache entries expire after 3 days


class ResponseCache:
    """Local disk cache for Claude API responses."""

    def __init__(
        self,
        cache_dir: str | None = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._stats_file = self.cache_dir / "_stats.json"
        self._stats = self._load_stats()

    def _make_key(self, prompt: str, model: str = "", context: str = "") -> str:
        """Create a cache key from the prompt + optional context."""
        raw = f"{model}:{context}:{prompt}".strip().lower()
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_stats(self) -> dict[str, int]:
        if self._stats_file.exists():
            return json.loads(self._stats_file.read_text())
        return {"hits": 0, "misses": 0, "total_saved_tokens": 0}

    def _save_stats(self):
        self._stats_file.write_text(json.dumps(self._stats, indent=2))

    def get(
        self,
        prompt: str,
        model: str = "",
        context: str = "",
    ) -> dict[str, Any] | None:
        """Look up a cached response. Returns None on miss."""
        key = self._make_key(prompt, model, context)
        path = self._cache_path(key)

        if not path.exists():
            self._stats["misses"] += 1
            self._save_stats()
            return None

        entry = json.loads(path.read_text())

        # Check TTL
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if datetime.now() - cached_at > self.ttl:
            path.unlink()
            self._stats["misses"] += 1
            self._save_stats()
            return None

        self._stats["hits"] += 1
        self._stats["total_saved_tokens"] += entry.get("tokens", 0)
        self._save_stats()

        return {
            "response": entry["response"],
            "tokens": entry.get("tokens", 0),
            "cached_at": entry["cached_at"],
            "source": "cache",
        }

    def set(
        self,
        prompt: str,
        response: str,
        model: str = "",
        context: str = "",
        tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        """Store a response in the cache."""
        key = self._make_key(prompt, model, context)
        entry = {
            "prompt": prompt[:200],  # Store truncated prompt for debugging
            "response": response,
            "model": model,
            "tokens": tokens,
            "cached_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._cache_path(key).write_text(json.dumps(entry, indent=2))

    def get_or_call(
        self,
        prompt: str,
        api_call_fn,
        model: str = "",
        context: str = "",
    ) -> dict[str, Any]:
        """Check cache first, call API only on miss. Saves the result."""
        cached = self.get(prompt, model, context)
        if cached:
            return cached

        # Cache miss - call the API
        result = api_call_fn(prompt)
        response_text = result if isinstance(result, str) else str(result)
        self.set(prompt, response_text, model, context)
        return {"response": response_text, "source": "api"}

    def invalidate(self, prompt: str, model: str = "", context: str = ""):
        """Remove a specific entry from cache."""
        key = self._make_key(prompt, model, context)
        path = self._cache_path(key)
        if path.exists():
            path.unlink()

    def clear(self):
        """Clear entire cache."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            if f.name != "_stats.json":
                f.unlink()
                count += 1
        self._stats = {"hits": 0, "misses": 0, "total_saved_tokens": 0}
        self._save_stats()
        return {"cleared": count}

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        removed = 0
        for f in self.cache_dir.glob("*.json"):
            if f.name == "_stats.json":
                continue
            try:
                entry = json.loads(f.read_text())
                cached_at = datetime.fromisoformat(entry["cached_at"])
                if datetime.now() - cached_at > self.ttl:
                    f.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError):
                f.unlink()
                removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        """Show cache performance stats."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0

        # Count current cache entries
        entries = list(self.cache_dir.glob("*.json"))
        entry_count = len([e for e in entries if e.name != "_stats.json"])

        # Estimate dollar savings (using Sonnet pricing as default)
        saved_tokens = self._stats["total_saved_tokens"]
        savings = (saved_tokens / 1_000_000) * 3.00  # Sonnet input price

        return {
            "cache_entries": entry_count,
            "total_lookups": total,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "tokens_saved": saved_tokens,
            "estimated_savings": f"${savings:.4f}",
        }

    def list_entries(self) -> list[dict[str, Any]]:
        """List all cached entries with metadata."""
        entries = []
        for f in self.cache_dir.glob("*.json"):
            if f.name == "_stats.json":
                continue
            try:
                data = json.loads(f.read_text())
                entries.append({
                    "key": f.stem,
                    "prompt_preview": data.get("prompt", "")[:80],
                    "model": data.get("model", ""),
                    "tokens": data.get("tokens", 0),
                    "cached_at": data.get("cached_at", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        entries.sort(key=lambda e: e.get("cached_at", ""), reverse=True)
        return entries


def main():
    import sys

    cache = ResponseCache()

    if len(sys.argv) < 2 or sys.argv[1] == "--stats":
        print(json.dumps(cache.stats(), indent=2))
    elif sys.argv[1] == "--list":
        for entry in cache.list_entries():
            print(f"  {entry['key']}  {entry['prompt_preview']}")
    elif sys.argv[1] == "--clear":
        result = cache.clear()
        print(f"Cleared {result['cleared']} cache entries.")
    elif sys.argv[1] == "--cleanup":
        removed = cache.cleanup_expired()
        print(f"Removed {removed} expired entries.")
    elif sys.argv[1] == "--get":
        result = cache.get(" ".join(sys.argv[2:]))
        if result:
            print(f"HIT: {result['response'][:200]}...")
        else:
            print("MISS: Not in cache.")
    else:
        print("Usage:")
        print("  python -m tools.response_cache --stats")
        print("  python -m tools.response_cache --list")
        print("  python -m tools.response_cache --clear")
        print("  python -m tools.response_cache --cleanup")
        print("  python -m tools.response_cache --get 'prompt text'")


if __name__ == "__main__":
    main()
