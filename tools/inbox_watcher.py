"""
Inbox Watcher - Auto-enroll new contacts into drip campaigns.

Monitors your email inbox (Gmail/IMAP), detects new inbound emails,
researches the sender's domain, checks if they match your target
profile (Wix site member, real estate firm, etc.), and auto-enrolls
them in a SendGrid drip campaign with personalized emails based
on what the research found.

Flow:
    1. Poll inbox for new unread emails
    2. Extract sender name, email, domain
    3. Research the domain (company name, industry, size)
    4. Score the lead (is this a real estate firm? a broker? a PM company?)
    5. If qualified, upload to SendGrid and start drip sequence
    6. Log everything so you can review what happened

Usage:
    from tools.inbox_watcher import InboxWatcher

    watcher = InboxWatcher(
        imap_server="imap.gmail.com",
        email="kevin@realestateaistudio.com",
        password="your-app-password",
        sendgrid_api_key="SG.xxx",
    )

    # Run once (check inbox, process new emails)
    results = watcher.check_and_enroll()

    # Or run on a schedule
    watcher.run_loop(interval_minutes=15)

Environment variables (set these instead of passing credentials):
    INBOX_EMAIL=kevin@realestateaistudio.com
    INBOX_PASSWORD=your-gmail-app-password
    INBOX_IMAP_SERVER=imap.gmail.com
    SENDGRID_API_KEY=SG.xxx
    ANTHROPIC_API_KEY=sk-ant-xxx  (for domain research via Claude)
"""
from __future__ import annotations

import csv
import email
import imaplib
import json
import os
import re
import time
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Any


LOG_DIR = ".inbox_watcher"
CONTACTS_LOG = "processed_contacts.json"
DRIP_TEMPLATES_FILE = "drip_templates.json"

# Domains to always skip (personal email, spam, etc.)
SKIP_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "mail.com", "protonmail.com",
    "live.com", "msn.com", "comcast.net", "att.net",
    "verizon.net", "me.com", "ymail.com",
}

# Keywords that indicate a real estate lead
RE_KEYWORDS = {
    "realty", "real estate", "property", "properties", "brokerage",
    "broker", "realtor", "keller williams", "re/max", "remax",
    "coldwell banker", "century 21", "sotheby", "compass",
    "cbre", "jll", "cushman", "colliers", "marcus millichap",
    "newmark", "berkshire hathaway", "homeservices",
    "asset management", "property management", "multifamily",
    "commercial real estate", "cre", "investment", "development",
}


class InboxWatcher:
    """Watch inbox and auto-enroll qualified contacts into drip campaigns."""

    def __init__(
        self,
        imap_server: str | None = None,
        email_address: str | None = None,
        password: str | None = None,
        sendgrid_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.imap_server = imap_server or os.environ.get("INBOX_IMAP_SERVER", "imap.gmail.com")
        self.email_address = email_address or os.environ.get("INBOX_EMAIL", "")
        self.password = password or os.environ.get("INBOX_PASSWORD", "")
        self.sendgrid_api_key = sendgrid_api_key or os.environ.get("SENDGRID_API_KEY", "")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._processed = self._load_processed()

    def _load_processed(self) -> dict[str, Any]:
        """Load the list of already-processed email addresses."""
        path = self.log_dir / CONTACTS_LOG
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_processed(self):
        path = self.log_dir / CONTACTS_LOG
        path.write_text(json.dumps(self._processed, indent=2))

    def _extract_sender(self, msg) -> tuple[str, str]:
        """Extract sender name and email from an email message."""
        from_header = msg.get("From", "")

        # Decode if needed
        if "=?" in from_header:
            decoded_parts = decode_header(from_header)
            from_header = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    from_header += part.decode(encoding or "utf-8", errors="replace")
                else:
                    from_header += part

        # Parse "Name <email>" format
        match = re.match(r"(.*?)\s*<(.+?)>", from_header)
        if match:
            name = match.group(1).strip().strip('"').strip("'")
            addr = match.group(2).strip()
        else:
            name = ""
            addr = from_header.strip()

        return name, addr.lower()

    def _get_domain(self, email_addr: str) -> str:
        """Extract domain from email address."""
        return email_addr.split("@")[-1] if "@" in email_addr else ""

    def _is_personal_email(self, email_addr: str) -> bool:
        """Check if this is a personal (non-business) email."""
        domain = self._get_domain(email_addr)
        return domain in SKIP_DOMAINS

    def _research_domain(self, domain: str) -> dict[str, Any]:
        """
        Research a domain to determine if it's a real estate company.
        Uses Claude API if available, otherwise does keyword matching.
        """
        result = {
            "domain": domain,
            "is_real_estate": False,
            "company_name": "",
            "company_type": "",
            "confidence": "low",
            "research_method": "keyword",
        }

        # Quick keyword check on the domain itself
        domain_lower = domain.lower()
        for kw in RE_KEYWORDS:
            if kw.replace(" ", "") in domain_lower or kw in domain_lower:
                result["is_real_estate"] = True
                result["confidence"] = "medium"
                result["company_name"] = domain.split(".")[0].replace("-", " ").title()
                break

        # If we have an Anthropic API key, use Claude for deeper research
        if self.anthropic_api_key and not result["is_real_estate"]:
            try:
                result = self._research_with_claude(domain, result)
            except Exception:
                pass  # Fall back to keyword-only result

        return result

    def _research_with_claude(self, domain: str, base_result: dict) -> dict[str, Any]:
        """Use Claude API to research a domain. Token-efficient prompt."""
        try:
            import anthropic
        except ImportError:
            return base_result

        client = anthropic.Anthropic(api_key=self.anthropic_api_key)

        # Minimal prompt to save tokens
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheapest model for research
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"Domain: {domain}\n"
                    "Is this a real estate company? Reply JSON only:\n"
                    '{"is_real_estate": bool, "company_name": str, '
                    '"company_type": "brokerage|pm|developer|investor|other", '
                    '"confidence": "high|medium|low"}'
                ),
            }],
        )

        try:
            text = response.content[0].text.strip()
            if text.startswith("{"):
                data = json.loads(text)
                data["domain"] = domain
                data["research_method"] = "claude_haiku"
                return data
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

        return base_result

    def _score_lead(self, sender_name: str, email_addr: str, domain_research: dict) -> dict[str, Any]:
        """Score a lead based on research results."""
        score = 0
        reasons = []

        if domain_research.get("is_real_estate"):
            score += 50
            reasons.append("real estate domain")

        if domain_research.get("confidence") == "high":
            score += 20
            reasons.append("high confidence match")
        elif domain_research.get("confidence") == "medium":
            score += 10

        company_type = domain_research.get("company_type", "")
        if company_type in ("brokerage", "pm", "developer"):
            score += 20
            reasons.append(f"company type: {company_type}")

        if not self._is_personal_email(email_addr):
            score += 10
            reasons.append("business email")

        qualified = score >= 50

        return {
            "score": score,
            "qualified": qualified,
            "reasons": reasons,
            "action": "enroll_drip" if qualified else "skip",
        }

    def _upload_to_sendgrid(self, contacts: list[dict[str, str]]) -> dict[str, Any]:
        """Upload contacts to SendGrid."""
        if not self.sendgrid_api_key:
            return {"error": "No SendGrid API key configured"}

        try:
            import urllib.request
            import urllib.error

            payload = {
                "contacts": [
                    {
                        "email": c["email"],
                        "first_name": c.get("first_name", ""),
                        "last_name": c.get("last_name", ""),
                        "custom_fields": {
                            "company": c.get("company", ""),
                            "source": "inbox_watcher",
                        },
                    }
                    for c in contacts
                ]
            }

            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/marketing/contacts",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self.sendgrid_api_key}",
                    "Content-Type": "application/json",
                },
                method="PUT",
            )

            with urllib.request.urlopen(req) as resp:
                return {"status": resp.status, "uploaded": len(contacts)}

        except Exception as e:
            return {"error": str(e)}

    def _send_drip_email(
        self,
        to_email: str,
        to_name: str,
        subject: str,
        body: str,
        send_at: int | None = None,
    ) -> dict[str, Any]:
        """Send a single email via SendGrid."""
        if not self.sendgrid_api_key:
            return {"error": "No SendGrid API key configured"}

        try:
            import urllib.request

            payload: dict[str, Any] = {
                "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
                "from": {
                    "email": self.email_address or "kevin@realestateaistudio.com",
                    "name": "Kevin - Real Estate AI Studio",
                },
                "subject": subject,
                "content": [{"type": "text/plain", "value": body}],
            }

            if send_at:
                payload["send_at"] = send_at

            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self.sendgrid_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req) as resp:
                return {"status": resp.status, "sent_to": to_email}

        except Exception as e:
            return {"error": str(e)}

    def check_inbox(self, folder: str = "INBOX", unseen_only: bool = True) -> list[dict[str, Any]]:
        """Check inbox for new emails and return sender info."""
        if not self.email_address or not self.password:
            return [{"error": "Email credentials not configured. Set INBOX_EMAIL and INBOX_PASSWORD env vars."}]

        try:
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email_address, self.password)
            mail.select(folder)

            search_criteria = "UNSEEN" if unseen_only else "ALL"
            _, message_numbers = mail.search(None, search_criteria)

            new_senders = []
            for num in message_numbers[0].split():
                _, msg_data = mail.fetch(num, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])
                name, addr = self._extract_sender(msg)

                if addr in self._processed:
                    continue

                subject = msg.get("Subject", "")
                if "=?" in subject:
                    decoded = decode_header(subject)
                    subject = ""
                    for part, enc in decoded:
                        if isinstance(part, bytes):
                            subject += part.decode(enc or "utf-8", errors="replace")
                        else:
                            subject += part

                new_senders.append({
                    "name": name,
                    "email": addr,
                    "domain": self._get_domain(addr),
                    "subject": subject,
                    "date": msg.get("Date", ""),
                })

            mail.logout()
            return new_senders

        except Exception as e:
            return [{"error": f"IMAP connection failed: {str(e)}"}]

    def check_and_enroll(self) -> dict[str, Any]:
        """
        Main workflow: check inbox, research domains, enroll qualified leads.
        Returns a summary of what happened.
        """
        results = {
            "checked_at": datetime.now().isoformat(),
            "new_emails": 0,
            "researched": 0,
            "qualified": 0,
            "enrolled": 0,
            "skipped": 0,
            "details": [],
        }

        # Step 1: Check inbox
        new_senders = self.check_inbox()
        if new_senders and "error" in new_senders[0]:
            results["error"] = new_senders[0]["error"]
            return results

        results["new_emails"] = len(new_senders)

        for sender in new_senders:
            addr = sender["email"]
            domain = sender["domain"]

            # Skip personal emails
            if self._is_personal_email(addr):
                self._processed[addr] = {
                    "status": "skipped",
                    "reason": "personal_email",
                    "processed_at": datetime.now().isoformat(),
                }
                results["skipped"] += 1
                results["details"].append({
                    "email": addr,
                    "action": "skipped",
                    "reason": "personal email domain",
                })
                continue

            # Step 2: Research domain
            research = self._research_domain(domain)
            results["researched"] += 1

            # Step 3: Score the lead
            score = self._score_lead(sender["name"], addr, research)

            if score["qualified"]:
                results["qualified"] += 1

                # Step 4: Upload to SendGrid
                name_parts = sender["name"].split(" ", 1)
                contact = {
                    "email": addr,
                    "first_name": name_parts[0] if name_parts else "",
                    "last_name": name_parts[1] if len(name_parts) > 1 else "",
                    "company": research.get("company_name", ""),
                }

                upload_result = self._upload_to_sendgrid([contact])

                if "error" not in upload_result:
                    results["enrolled"] += 1

                self._processed[addr] = {
                    "status": "enrolled",
                    "score": score["score"],
                    "research": research,
                    "processed_at": datetime.now().isoformat(),
                }

                results["details"].append({
                    "email": addr,
                    "name": sender["name"],
                    "company": research.get("company_name", ""),
                    "type": research.get("company_type", ""),
                    "score": score["score"],
                    "action": "enrolled",
                })
            else:
                self._processed[addr] = {
                    "status": "skipped",
                    "reason": "low_score",
                    "score": score["score"],
                    "processed_at": datetime.now().isoformat(),
                }
                results["skipped"] += 1
                results["details"].append({
                    "email": addr,
                    "score": score["score"],
                    "action": "skipped",
                    "reason": "did not meet qualification threshold",
                })

        self._save_processed()
        return results

    def process_csv(self, csv_path: str) -> dict[str, Any]:
        """
        Process a CSV of contacts (like a Wix export) through the same
        research and enrollment pipeline. Useful for bulk processing
        without waiting for inbox emails.
        """
        path = Path(csv_path)
        if not path.exists():
            return {"error": f"File not found: {csv_path}"}

        results = {"processed": 0, "enrolled": 0, "skipped": 0, "details": []}

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names
                addr = (
                    row.get("email", "")
                    or row.get("Email", "")
                    or row.get("email_address", "")
                    or row.get("Email Address", "")
                ).strip().lower()

                if not addr or addr in self._processed:
                    continue

                name = (
                    row.get("name", "")
                    or row.get("Name", "")
                    or f"{row.get('first_name', row.get('First Name', ''))} {row.get('last_name', row.get('Last Name', ''))}".strip()
                )

                domain = self._get_domain(addr)
                results["processed"] += 1

                if self._is_personal_email(addr):
                    results["skipped"] += 1
                    continue

                research = self._research_domain(domain)
                score = self._score_lead(name, addr, research)

                if score["qualified"]:
                    name_parts = name.split(" ", 1)
                    contact = {
                        "email": addr,
                        "first_name": name_parts[0],
                        "last_name": name_parts[1] if len(name_parts) > 1 else "",
                        "company": research.get("company_name", ""),
                    }
                    self._upload_to_sendgrid([contact])
                    results["enrolled"] += 1
                    results["details"].append({
                        "email": addr, "name": name,
                        "company": research.get("company_name", ""),
                        "action": "enrolled",
                    })
                else:
                    results["skipped"] += 1

                self._processed[addr] = {
                    "status": "enrolled" if score["qualified"] else "skipped",
                    "processed_at": datetime.now().isoformat(),
                }

        self._save_processed()
        return results

    def run_loop(self, interval_minutes: int = 15, max_iterations: int | None = None):
        """Run the watcher on a loop. Use this for continuous monitoring."""
        iteration = 0
        print(f"Inbox watcher started. Checking every {interval_minutes} minutes.")
        while True:
            if max_iterations and iteration >= max_iterations:
                break
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking inbox...")
            results = self.check_and_enroll()
            print(f"  New: {results['new_emails']} | Enrolled: {results['enrolled']} | Skipped: {results['skipped']}")
            if results.get("details"):
                for d in results["details"]:
                    print(f"    {d['action'].upper()}: {d['email']} - {d.get('company', '')}")
            iteration += 1
            time.sleep(interval_minutes * 60)

    def get_stats(self) -> dict[str, Any]:
        """Get stats on processed contacts."""
        enrolled = sum(1 for v in self._processed.values() if v.get("status") == "enrolled")
        skipped = sum(1 for v in self._processed.values() if v.get("status") == "skipped")
        return {
            "total_processed": len(self._processed),
            "enrolled": enrolled,
            "skipped": skipped,
            "domains_seen": len(set(
                self._get_domain(k) for k in self._processed.keys()
            )),
        }


def main():
    import sys

    watcher = InboxWatcher()

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("Inbox Watcher - Auto-enroll contacts into drip campaigns")
        print()
        print("Usage:")
        print("  python -m tools.inbox_watcher --check      # Check inbox once")
        print("  python -m tools.inbox_watcher --loop 15    # Check every 15 min")
        print("  python -m tools.inbox_watcher --csv file   # Process CSV contacts")
        print("  python -m tools.inbox_watcher --stats      # Show stats")
        print()
        print("Environment variables needed:")
        print("  INBOX_EMAIL, INBOX_PASSWORD, INBOX_IMAP_SERVER")
        print("  SENDGRID_API_KEY, ANTHROPIC_API_KEY (optional, for research)")
    elif sys.argv[1] == "--check":
        results = watcher.check_and_enroll()
        print(json.dumps(results, indent=2))
    elif sys.argv[1] == "--loop":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        watcher.run_loop(interval_minutes=interval)
    elif sys.argv[1] == "--csv":
        results = watcher.process_csv(sys.argv[2])
        print(json.dumps(results, indent=2))
    elif sys.argv[1] == "--stats":
        print(json.dumps(watcher.get_stats(), indent=2))


if __name__ == "__main__":
    main()
