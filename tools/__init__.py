"""
Token Optimization Toolkit for Claude Code
============================================
Python tools that reduce token/credit consumption when working with Claude.
Built for Real Estate AI Studio operations.

Tools:
- token_counter: Estimate tokens and cost before making API calls
- response_cache: Cache Claude responses to avoid paying for duplicate queries
- context_pruner: Strip unnecessary content from files before sending to Claude
- prompt_templates: Reusable prompt templates with variable substitution
- batch_processor: Combine multiple queries into efficient batched API calls
- file_differ: Send only changed lines instead of full files
"""

__version__ = "1.0.0"
