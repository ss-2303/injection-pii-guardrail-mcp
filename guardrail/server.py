"""
Agent Guardrail MCP Server.

Exposes the injection detector, PII/secrets detector, and audit log as
MCP tools that any MCP-compatible agent (Claude Desktop, Claude Code,
custom pipelines) can call.

Tools exposed:
- scan_input(text, source)  — check incoming text for prompt injection
- scan_output(text)         — check outgoing text for PII/secrets, get a redacted version
- get_audit_trail(limit, risk_level) — query past scan results
- get_guardrail_stats()     — aggregate stats across all scans

Each scan tool follows the same pattern:
  1. Run the relevant detector
  2. Log the result to the audit trail
  3. Return a structured dict with score/findings/recommendation

Run standalone for local testing:
    python3 server.py

Connect to Claude Desktop by adding to claude_desktop_config.json (see README).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .injection_detector import scan_text
from .pii_detector import scan_and_redact
from .audit import log_scan, get_audit_trail as _get_audit_trail, get_stats as _get_stats


mcp = FastMCP("agent-guardrail")


# ─── Helpers: dataclass → dict conversion ───────────────────────────
#
# MCP tools return JSON-serializable data. Our detectors return
# dataclasses, so each tool function converts the result into a plain
# dict before returning it.


def _injection_recommendation(risk_level: str) -> str:
    """Map injection risk level to a recommendation string."""
    return {"low": "Proceed", "medium": "Flag for review", "high": "Block"}[risk_level]


def _pii_recommendation(risk_level: str) -> str:
    """Map PII risk level to a recommendation string."""
    return {"low": "Proceed", "medium": "Flag for review", "high": "Block"}[risk_level]


# ─── Tools ───────────────────────────────────────────────────────────


@mcp.tool()
def scan_input(text: str, source: str = "unspecified") -> dict:
    """
    Scan incoming text for prompt injection attempts.

    Use this before an agent acts on user input, retrieved documents,
    tool outputs, or any other text that could contain hidden instructions.

    Args:
        text: The text to scan for injection patterns.
        source: Where this text came from (e.g. "user_input", "document_content",
            "tool_output"). Recorded in the audit trail for traceability.

    Returns:
        A dict with: score (0-100), risk_level (low/medium/high),
        reasons (list of matched pattern explanations), and recommendation
        (Proceed / Flag for review / Block).
    """
    result = scan_text(text)
    recommendation = _injection_recommendation(result.risk_level)
    reasons = [m.reason for m in result.matches]

    log_scan(
        scan_type="injection",
        source=source,
        risk_level=result.risk_level,
        risk_score=result.score,
        reasons=reasons,
        recommendation=recommendation,
        text=text,
    )

    return {
        "score": result.score,
        "risk_level": result.risk_level,
        "reasons": reasons,
        "matches": [
            {"pattern": m.pattern_name, "matched_text": m.matched_text, "weight": m.weight}
            for m in result.matches
        ],
        "recommendation": recommendation,
    }


@mcp.tool()
def scan_output(text: str) -> dict:
    """
    Scan outgoing text for PII and secrets/credentials before an agent sends it.

    Use this on agent-generated responses before returning them to a user,
    posting them externally, or logging them anywhere outside this system.

    Args:
        text: The text to scan for PII (emails, phone numbers, SSNs, credit
            cards) and secrets (AWS keys, GitHub tokens, API keys, private keys).

    Returns:
        A dict with: risk_level (low/medium/high), findings (list of what was
        detected and where), redacted_text (safe version with sensitive data
        replaced by labeled tokens), and recommendation.
    """
    result = scan_and_redact(text)
    recommendation = _pii_recommendation(result.risk_level)
    reasons = [f.finding_type for f in result.findings]

    log_scan(
        scan_type="pii",
        source="agent_output",
        risk_level=result.risk_level,
        risk_score=0,  # PII detector doesn't use a numeric score, only risk_level
        reasons=reasons,
        recommendation=recommendation,
        text=text,
    )

    return {
        "risk_level": result.risk_level,
        "finding_count": result.finding_count,
        "findings": [
            {"type": f.finding_type, "matched_text": f.matched_text, "severity": f.severity}
            for f in result.findings
        ],
        "redacted_text": result.redacted_text,
        "recommendation": recommendation,
    }


@mcp.tool()
def get_audit_trail(limit: int = 20, risk_level: str | None = None) -> list[dict]:
    """
    Retrieve recent entries from the guardrail audit log.

    Use this to review what scans have been performed, check compliance
    history, or investigate flagged activity.

    Args:
        limit: Maximum number of entries to return (most recent first).
        risk_level: Optional filter — only return entries matching this
            risk level ("low", "medium", or "high").

    Returns:
        A list of audit entries, each with id, timestamp, scan_type, source,
        risk_level, risk_score, reasons, recommendation, and a text preview
        (never the full scanned text).
    """
    entries = _get_audit_trail(limit=limit, risk_level=risk_level)
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp,
            "scan_type": e.scan_type,
            "source": e.source,
            "risk_level": e.risk_level,
            "risk_score": e.risk_score,
            "reasons": e.reasons,
            "recommendation": e.recommendation,
            "text_preview": e.text_preview,
        }
        for e in entries
    ]


@mcp.tool()
def get_guardrail_stats() -> dict:
    """
    Get aggregate statistics across all scans performed by the guardrail system.

    Use this for a dashboard-style overview: how many scans total, broken
    down by risk level, scan type, and recommendation.

    Returns:
        A dict with total_scans, by_risk_level, by_scan_type, and
        by_recommendation breakdowns.
    """
    stats = _get_stats()
    return {
        "total_scans": stats.total_scans,
        "by_risk_level": stats.by_risk_level,
        "by_scan_type": stats.by_scan_type,
        "by_recommendation": stats.by_recommendation,
    }


def main() -> None:
    """Entry point for the `agent-guardrail-mcp` console script."""
    mcp.run()


if __name__ == "__main__":
    main()