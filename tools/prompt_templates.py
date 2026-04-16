"""
Prompt Templates - Stop rewriting the same prompts.

Pre-built, reusable prompt templates with variable substitution.
Instead of writing a 500-token prompt from scratch every time,
load a 50-token template and fill in the blanks. Saves tokens
on every single API call.

Includes templates for your four projects:
  - Sales Engine (outreach, research, call prep)
  - Content/Brand (LinkedIn, email sequences, SEO)
  - Product Dev (specs, architecture, sprint planning)
  - Ops/Strategy (pipeline, forecasts, status reports)

Usage:
    from tools.prompt_templates import TemplateManager

    tm = TemplateManager()

    # Use a built-in template
    prompt = tm.render("cold_email", {
        "prospect_name": "John Smith",
        "company": "Keller Williams Vegas",
        "pain_point": "junior brokers drowning in admin work",
    })

    # Create your own template
    tm.save("my_template", "Write a {doc_type} about {topic} for {audience}")
    prompt = tm.render("my_template", {"doc_type": "guide", "topic": "AI", "audience": "brokers"})
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATES_DIR = ".prompt_templates"

# Built-in templates organized by project
BUILTIN_TEMPLATES = {
    # ── Sales Engine ──
    "cold_email": (
        "Write a cold outreach email from Real Estate AI Studio to {prospect_name} "
        "at {company}. Their pain point: {pain_point}. "
        "Position our managed AI services as the solution. "
        "Keep it under 150 words. No em dashes. Conversational tone. "
        "End with a soft CTA for a 15-minute call."
    ),
    "follow_up_email": (
        "Write a follow-up email to {prospect_name} at {company}. "
        "Original email was about {original_topic}. "
        "This is follow-up #{follow_up_number}. "
        "Keep it shorter than the original. No em dashes. "
        "Add one new insight about {value_add}."
    ),
    "breakup_email": (
        "Write a final breakup email to {prospect_name} at {company}. "
        "We reached out about {topic} with no response. "
        "Keep it to 3-4 sentences. Acknowledge the silence, leave the door open. "
        "No em dashes. No guilt trip."
    ),
    "account_research": (
        "Research {company} in {market}. I need: "
        "1) Company size and focus area "
        "2) Key decision makers (names + titles) "
        "3) Current tech stack if visible "
        "4) Pain points AI could solve for them "
        "5) Best angle to pitch managed AI services. "
        "Be specific and actionable."
    ),
    "call_prep": (
        "Prepare me for a call with {prospect_name}, {title} at {company}. "
        "They responded to our email about {topic}. "
        "Give me: opening line, 3 discovery questions, "
        "2 objection handlers, and a close for next steps. "
        "Keep each item to one sentence."
    ),
    "call_summary": (
        "Summarize this sales call and extract action items. "
        "Prospect: {prospect_name} at {company}. "
        "Call notes: {notes}. "
        "Output: 1) Key takeaways 2) Objections raised "
        "3) Next steps with deadlines 4) Deal probability (hot/warm/cold)."
    ),
    # ── Content & Brand ──
    "linkedin_post": (
        "Write a LinkedIn post for Real Estate AI Studio about {topic}. "
        "Target audience: {audience}. "
        "Hook in the first line. Include a specific example or stat. "
        "End with a question to drive engagement. "
        "Under 200 words. No em dashes. No hashtag spam (max 3)."
    ),
    "blog_post_outline": (
        "Create an SEO-optimized blog post outline about {topic}. "
        "Target keyword: {keyword}. "
        "Audience: {audience}. "
        "Include: title, meta description, H2 headers, "
        "key points under each header, CTA at the end. "
        "Make it actionable, not fluffy."
    ),
    "email_sequence": (
        "Design a {num_emails}-email drip sequence for {audience}. "
        "Goal: {goal}. "
        "For each email give: subject line, send timing, "
        "key message (one sentence), CTA. "
        "No em dashes."
    ),
    "social_calendar": (
        "Create a {num_weeks}-week social media calendar for Real Estate AI Studio. "
        "Platforms: {platforms}. "
        "Themes: {themes}. "
        "For each post: platform, day, topic, hook line, content type (text/image/video)."
    ),
    # ── Product Dev ──
    "write_spec": (
        "Write a product spec for {feature_name}. "
        "Context: {context}. "
        "Include: problem statement, proposed solution, "
        "user stories (max 5), acceptance criteria, "
        "technical considerations, out of scope."
    ),
    "system_design": (
        "Design the system architecture for {system_name}. "
        "Requirements: {requirements}. "
        "Include: components, data flow, tech stack recommendation, "
        "API endpoints, database schema, deployment approach."
    ),
    "code_task": (
        "Implement {task_description}. "
        "Language: {language}. "
        "Context: {context}. "
        "Requirements: {requirements}. "
        "Keep it production-ready. No unnecessary abstractions."
    ),
    # ── Ops & Strategy ──
    "pipeline_review": (
        "Review this sales pipeline and give recommendations. "
        "Pipeline data: {pipeline_data}. "
        "Identify: stale deals, highest probability closes, "
        "deals that need attention this week, revenue forecast for {timeframe}."
    ),
    "status_report": (
        "Generate a weekly status report for {week_ending}. "
        "Accomplishments: {accomplishments}. "
        "In progress: {in_progress}. "
        "Blockers: {blockers}. "
        "Format: executive summary (3 bullets), details, next week priorities."
    ),
    "competitive_intel": (
        "Analyze competitor {competitor} in the {market} space. "
        "Compare to Real Estate AI Studio on: "
        "pricing, service model, target customer, strengths, weaknesses. "
        "End with 3 ways we can differentiate."
    ),
}


class TemplateManager:
    """Manage and render prompt templates."""

    def __init__(self, templates_dir: str | None = None):
        self.templates_dir = Path(templates_dir or DEFAULT_TEMPLATES_DIR)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._custom_templates: dict[str, str] = {}
        self._load_custom_templates()

    def _load_custom_templates(self):
        """Load user-created templates from disk."""
        for f in self.templates_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self._custom_templates[f.stem] = data["template"]
            except (json.JSONDecodeError, KeyError):
                continue

    def render(self, template_name: str, variables: dict[str, str]) -> str:
        """Render a template with variables. Returns the final prompt."""
        # Check custom templates first, then builtins
        if template_name in self._custom_templates:
            template = self._custom_templates[template_name]
        elif template_name in BUILTIN_TEMPLATES:
            template = BUILTIN_TEMPLATES[template_name]
        else:
            available = self.list_templates()
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {', '.join(available)}"
            )

        try:
            return template.format(**variables)
        except KeyError as e:
            # Show which variables are needed
            import re
            required = re.findall(r"\{(\w+)\}", template)
            provided = list(variables.keys())
            missing = [r for r in required if r not in provided]
            raise ValueError(
                f"Missing variables for '{template_name}': {missing}. "
                f"Required: {required}"
            ) from e

    def save(self, name: str, template: str, description: str = ""):
        """Save a custom template."""
        data = {
            "template": template,
            "description": description,
        }
        path = self.templates_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        self._custom_templates[name] = template

    def delete(self, name: str):
        """Delete a custom template."""
        path = self.templates_dir / f"{name}.json"
        if path.exists():
            path.unlink()
        self._custom_templates.pop(name, None)

    def list_templates(self) -> list[str]:
        """List all available template names."""
        return sorted(set(list(BUILTIN_TEMPLATES.keys()) + list(self._custom_templates.keys())))

    def list_by_project(self) -> dict[str, list[str]]:
        """Group templates by project area."""
        return {
            "Sales Engine": [
                "cold_email", "follow_up_email", "breakup_email",
                "account_research", "call_prep", "call_summary",
            ],
            "Content & Brand": [
                "linkedin_post", "blog_post_outline",
                "email_sequence", "social_calendar",
            ],
            "Product Dev": [
                "write_spec", "system_design", "code_task",
            ],
            "Ops & Strategy": [
                "pipeline_review", "status_report", "competitive_intel",
            ],
            "Custom": list(self._custom_templates.keys()),
        }

    def preview(self, template_name: str) -> dict[str, Any]:
        """Preview a template with its required variables."""
        import re

        if template_name in self._custom_templates:
            template = self._custom_templates[template_name]
        elif template_name in BUILTIN_TEMPLATES:
            template = BUILTIN_TEMPLATES[template_name]
        else:
            raise ValueError(f"Template '{template_name}' not found.")

        variables = re.findall(r"\{(\w+)\}", template)
        tokens = int(len(template) / 4)

        return {
            "name": template_name,
            "template": template,
            "required_variables": variables,
            "template_tokens": tokens,
        }

    def estimate_savings(self, template_name: str) -> dict[str, Any]:
        """Estimate how many tokens you save by using this template vs writing freeform."""
        preview = self.preview(template_name)
        template_tokens = preview["template_tokens"]
        # Average freeform prompt for same task is roughly 3-5x longer
        estimated_freeform = template_tokens * 4
        savings = estimated_freeform - template_tokens

        return {
            "template_tokens": template_tokens,
            "estimated_freeform_tokens": estimated_freeform,
            "savings_per_use": savings,
            "savings_over_10_uses": savings * 10,
            "savings_over_100_uses": savings * 100,
        }


def main():
    import sys

    tm = TemplateManager()

    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        by_project = tm.list_by_project()
        for project, templates in by_project.items():
            if templates:
                print(f"\n{project}:")
                for t in templates:
                    print(f"  - {t}")
    elif sys.argv[1] == "--preview":
        name = sys.argv[2]
        result = tm.preview(name)
        print(f"Template: {result['name']}")
        print(f"Variables: {result['required_variables']}")
        print(f"Tokens: {result['template_tokens']}")
        print(f"\n{result['template']}")
    elif sys.argv[1] == "--render":
        name = sys.argv[2]
        # Parse key=value pairs
        variables = {}
        for arg in sys.argv[3:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                variables[k] = v
        print(tm.render(name, variables))
    else:
        print("Usage:")
        print("  python -m tools.prompt_templates --list")
        print("  python -m tools.prompt_templates --preview cold_email")
        print("  python -m tools.prompt_templates --render cold_email prospect_name=John company=KW")


if __name__ == "__main__":
    main()
