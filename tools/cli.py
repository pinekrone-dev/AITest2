"""
Token Optimization Toolkit - CLI Entry Point

Run any tool from the command line:
    python -m tools token-count "your prompt here"
    python -m tools prune app.py
    python -m tools cache --stats
    python -m tools diff app.py
    python -m tools templates --list
    python -m tools batch --demo
    python -m tools inbox --check
    python -m tools dashboard
"""
from __future__ import annotations

import json
import sys


USAGE = """
Token Optimization Toolkit for Claude Code
============================================

Commands:
  token-count <text>              Estimate tokens and cost for text
  token-count --file <path>       Estimate tokens for a file
  token-count --dir <path>        Estimate tokens for a directory
  token-count --compare <text>    Compare cost across all models

  prune <file>                    Prune file (strip comments, blanks, docstrings)
  prune <file> --skeleton         Extract function/class signatures only
  prune <file> --aggressive       Maximum pruning for code review

  cache --stats                   Show cache hit rate and savings
  cache --list                    List cached responses
  cache --clear                   Clear cache
  cache --get "prompt"            Look up a cached response

  diff <file>                     Diff file against last snapshot
  diff <file> --smart             Smart diff with function context
  diff <file> --snap              Take a snapshot (no diff)

  templates --list                List all prompt templates
  templates --preview <name>      Preview a template with its variables
  templates --render <name> k=v   Render a template with variables

  batch --demo                    Show batching demo and savings estimate

  inbox --check                   Check inbox and auto-enroll qualified leads
  inbox --csv <file>              Process a CSV of contacts
  inbox --loop <minutes>          Run inbox watcher on a loop
  inbox --stats                   Show inbox watcher stats

  dashboard                       Show overall token savings dashboard

Examples:
  python -m tools token-count "Write a cold email for Vegas brokers"
  python -m tools token-count --file app.py
  python -m tools prune app.py --skeleton
  python -m tools templates --render cold_email prospect_name=John company=KW
  python -m tools inbox --check
"""


def dashboard():
    """Show overall savings dashboard across all tools."""
    from tools.response_cache import ResponseCache
    from tools.file_differ import FileDiffer

    cache = ResponseCache()
    differ = FileDiffer()

    cache_stats = cache.stats()
    snapshots = differ.list_snapshots()

    print("=" * 50)
    print("  Token Optimization Dashboard")
    print("=" * 50)
    print()
    print("Response Cache:")
    print(f"  Entries:          {cache_stats['cache_entries']}")
    print(f"  Hit rate:         {cache_stats['hit_rate']}")
    print(f"  Tokens saved:     {cache_stats['tokens_saved']:,}")
    print(f"  Est. savings:     {cache_stats['estimated_savings']}")
    print()
    print("File Snapshots:")
    print(f"  Files tracked:    {len(snapshots)}")
    print()
    print("Available Templates:")

    from tools.prompt_templates import TemplateManager
    tm = TemplateManager()
    by_project = tm.list_by_project()
    for project, templates in by_project.items():
        if templates:
            print(f"  {project}: {len(templates)} templates")

    print()
    print("Quick wins:")
    print("  1. Use 'prune --skeleton' before asking Claude about large files")
    print("  2. Use 'diff' instead of sending full files after edits")
    print("  3. Use 'templates' instead of writing prompts from scratch")
    print("  4. Use 'batch' to combine multiple tasks into one API call")
    print("  5. Use 'cache' to avoid re-asking the same questions")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help"):
        print(USAGE)
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "token-count":
        from tools.token_counter import main as tc_main
        sys.argv = ["token_counter"] + args
        tc_main()

    elif command == "prune":
        from tools.context_pruner import main as cp_main
        sys.argv = ["context_pruner"] + args
        cp_main()

    elif command == "cache":
        from tools.response_cache import main as rc_main
        sys.argv = ["response_cache"] + args
        rc_main()

    elif command == "diff":
        from tools.file_differ import main as fd_main
        sys.argv = ["file_differ"] + args
        fd_main()

    elif command == "templates":
        from tools.prompt_templates import main as pt_main
        sys.argv = ["prompt_templates"] + args
        pt_main()

    elif command == "batch":
        from tools.batch_processor import main as bp_main
        sys.argv = ["batch_processor"] + args
        bp_main()

    elif command == "inbox":
        from tools.inbox_watcher import main as iw_main
        sys.argv = ["inbox_watcher"] + args
        iw_main()

    elif command == "dashboard":
        dashboard()

    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m tools --help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
