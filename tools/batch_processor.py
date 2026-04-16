"""
Batch Processor - One API call instead of ten.

Combines multiple independent queries into a single efficient
API call. Instead of making 10 separate requests (10x the overhead
tokens for system prompts, conversation context, etc.), batch them
into one structured request.

Also supports the Claude Batch API for async processing at 50% cost.

Usage:
    from tools.batch_processor import BatchProcessor

    bp = BatchProcessor()

    # Queue up tasks
    bp.add("email_vegas", "Write a cold email for Vegas brokerage XYZ")
    bp.add("email_phoenix", "Write a cold email for Phoenix PM firm ABC")
    bp.add("email_socal", "Write a cold email for SoCal developer DEF")

    # Generate a single combined prompt
    combined = bp.build_prompt()
    # Send this ONE prompt to Claude instead of 3 separate calls

    # Parse the structured response back into individual results
    results = bp.parse_response(claude_response)
    print(results["email_vegas"])

    # Or use the Batch API for 50% off
    batch_request = bp.build_batch_request(model="claude-sonnet-4-6")
"""
from __future__ import annotations

import json
import uuid
from typing import Any


class BatchProcessor:
    """Combine multiple queries into efficient batched calls."""

    def __init__(self):
        self._tasks: list[dict[str, str]] = []

    def add(self, task_id: str, prompt: str, context: str = ""):
        """Add a task to the batch."""
        self._tasks.append({
            "id": task_id,
            "prompt": prompt,
            "context": context,
        })

    def clear(self):
        """Clear all queued tasks."""
        self._tasks.clear()

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def build_prompt(self) -> str:
        """
        Combine all tasks into a single structured prompt.

        This is the key token saver: instead of repeating the system
        prompt and conversation context for each task, you send it once
        with all tasks bundled together.
        """
        if not self._tasks:
            return ""

        parts = [
            "Complete each of the following tasks. "
            "Return your responses in a JSON object where each key is the task_id "
            "and each value is your response for that task.",
            "",
            "Tasks:",
        ]

        for i, task in enumerate(self._tasks, 1):
            parts.append(f"\n--- Task {i}: {task['id']} ---")
            if task["context"]:
                parts.append(f"Context: {task['context']}")
            parts.append(f"Request: {task['prompt']}")

        parts.append(
            "\n\nReturn a JSON object with task IDs as keys. "
            "Example format: {\"task_id_1\": \"response 1\", \"task_id_2\": \"response 2\"}"
        )

        return "\n".join(parts)

    def build_batch_request(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """
        Build requests for the Claude Batch API (50% cost reduction).

        Returns a list of request objects ready to submit to
        POST /v1/messages/batches
        """
        requests = []
        for task in self._tasks:
            messages = [{"role": "user", "content": task["prompt"]}]
            if task["context"]:
                messages[0]["content"] = f"{task['context']}\n\n{task['prompt']}"

            request = {
                "custom_id": task["id"],
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                },
            }
            if system_prompt:
                request["params"]["system"] = system_prompt

            requests.append(request)

        return requests

    def parse_response(self, response_text: str) -> dict[str, str]:
        """Parse a combined response back into individual task results."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            elif "{" in response_text:
                # Find the outermost JSON object
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError):
            # If JSON parsing fails, try to split by task markers
            results = {}
            for task in self._tasks:
                marker = f"--- Task"
                if task["id"] in response_text:
                    # Try to extract the section
                    results[task["id"]] = f"[Parse error - raw response contains this task ID]"
                else:
                    results[task["id"]] = "[Could not parse response for this task]"
            results["_raw"] = response_text
            return results

    def estimate_savings(self) -> dict[str, Any]:
        """Estimate token savings from batching vs individual calls."""
        if not self._tasks:
            return {"error": "No tasks queued"}

        # Estimate overhead per individual call
        system_prompt_overhead = 200  # tokens for system prompt repeated each call
        conversation_overhead = 50   # tokens for message framing
        per_call_overhead = system_prompt_overhead + conversation_overhead

        individual_overhead = per_call_overhead * len(self._tasks)
        batched_overhead = per_call_overhead * 1  # Only pay overhead once

        # Task content tokens
        total_task_tokens = sum(
            int(len(t["prompt"]) / 4) + int(len(t.get("context", "")) / 4)
            for t in self._tasks
        )

        individual_total = individual_overhead + total_task_tokens
        batched_total = batched_overhead + total_task_tokens + 50  # 50 for batching instructions

        saved = individual_overhead - batched_overhead
        pct = (saved / individual_total * 100) if individual_total > 0 else 0

        return {
            "task_count": len(self._tasks),
            "individual_calls_tokens": individual_total,
            "batched_call_tokens": batched_total,
            "tokens_saved": saved,
            "savings": f"{pct:.0f}% overhead reduction",
            "batch_api_note": "Using the Batch API saves an additional 50% on per-token cost",
        }

    def preview(self) -> str:
        """Preview what the combined prompt will look like."""
        prompt = self.build_prompt()
        token_est = int(len(prompt) / 4)
        return f"Combined prompt ({token_est} tokens, {len(self._tasks)} tasks):\n\n{prompt}"


class EmailBatcher(BatchProcessor):
    """
    Specialized batcher for email campaigns.
    Queue up personalized emails and generate them all in one API call.
    """

    def add_email(
        self,
        recipient_name: str,
        company: str,
        email_type: str = "cold",
        custom_angle: str = "",
    ):
        """Add a personalized email to the batch."""
        task_id = f"email_{recipient_name.lower().replace(' ', '_')}"

        if email_type == "cold":
            prompt = (
                f"Write a cold outreach email to {recipient_name} at {company}. "
                f"From Real Estate AI Studio. "
                f"{'Angle: ' + custom_angle + '. ' if custom_angle else ''}"
                f"Under 150 words. No em dashes. Soft CTA for a 15-minute call."
            )
        elif email_type == "follow_up":
            prompt = (
                f"Write a follow-up email to {recipient_name} at {company}. "
                f"{'Focus: ' + custom_angle + '. ' if custom_angle else ''}"
                f"Shorter than the original. No em dashes. Add one new value point."
            )
        elif email_type == "breakup":
            prompt = (
                f"Write a breakup email to {recipient_name} at {company}. "
                f"3-4 sentences. Acknowledge silence. Leave door open. No em dashes."
            )
        else:
            prompt = f"Write a {email_type} email to {recipient_name} at {company}."

        self.add(task_id, prompt)

    def build_campaign_prompt(self) -> str:
        """Build an optimized prompt for email campaign generation."""
        if not self._tasks:
            return ""

        parts = [
            "You are writing emails for Real Estate AI Studio, a managed AI services "
            "firm for commercial real estate. Generate personalized emails for each "
            "recipient below. Rules: no em dashes ever, conversational tone, under "
            "150 words each, always include a soft CTA.",
            "",
            "Return a JSON object where keys are recipient identifiers and values "
            "are objects with 'subject' and 'body' fields.",
            "",
        ]

        for task in self._tasks:
            parts.append(f"Recipient: {task['id']}")
            parts.append(f"  {task['prompt']}")
            parts.append("")

        return "\n".join(parts)


def main():
    import sys

    bp = BatchProcessor()

    # Demo
    bp.add("task_1", "Write a one-paragraph summary of AI in real estate")
    bp.add("task_2", "List 5 pain points for commercial real estate brokers")
    bp.add("task_3", "Write a LinkedIn hook about AI property management")

    print("=== Combined Prompt ===")
    print(bp.build_prompt())
    print("\n=== Savings Estimate ===")
    print(json.dumps(bp.estimate_savings(), indent=2))


if __name__ == "__main__":
    main()
