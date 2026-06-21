"""
PII and secrets detector with redaction.

Scans text for personally identifiable information (emails, phone numbers,
SSNs, credit cards) and credentials/secrets (AWS keys, GitHub tokens,
API keys, PEM blocks). Returns findings with types and positions, and
produces a redacted version of the text safe for output.

Design decisions:
- Pattern-based, not ML — same rationale as the injection detector
- Redaction replaces matches with labeled tokens ([REDACTED_EMAIL], etc.)
  so the surrounding text remains useful
- Two severity tiers: credentials = high, PII-only = medium, nothing = low
- Each finding records the type and span (start/end positions) for audit
- Matches are processed longest-first to avoid overlapping redaction issues
"""

import re
from dataclasses import dataclass, field


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class Finding:
    """One piece of detected sensitive data."""
    finding_type: str       # e.g. "email", "aws_key", "ssn"
    matched_text: str       # The actual sensitive text found
    span: tuple[int, int]   # (start, end) position in the original text
    severity: str           # "pii" or "credential"


@dataclass
class PIIScanResult:
    """Complete result of scanning text for PII/secrets."""
    risk_level: str                             # "low", "medium", or "high"
    findings: list[Finding] = field(default_factory=list)
    redacted_text: str = ""                     # Text with sensitive data replaced
    finding_count: int = 0                      # Total number of findings


# ─── Detection Patterns ─────────────────────────────────────────────
#
# Each pattern: (compiled_regex, finding_type, redaction_label, severity)
#
# severity is either "pii" or "credential" — this drives the overall
# risk_level: any credential match = high, only PII = medium, none = low
#
# Order matters for overlapping patterns — more specific patterns
# (like AWS keys) should be checked, but since we process all patterns
# independently and then sort by span position, order here is just
# for readability.

PII_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [

    # ── Credentials (severity: credential → risk_level: high) ──

    # AWS Access Key IDs: always start with AKIA, followed by 16 alphanumeric chars
    (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "aws_access_key", "[REDACTED_AWS_KEY]", "credential"
    ),

    # AWS Secret Keys: 40 character base64-ish strings, usually after a label
    (
        re.compile(r"(?:aws_secret_access_key|secret_key)\s*[=:]\s*[A-Za-z0-9/+=]{40}"),
        "aws_secret_key", "[REDACTED_AWS_SECRET]", "credential"
    ),

    # GitHub personal access tokens: ghp_ followed by 36 alphanumeric chars
    (
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
        "github_token", "[REDACTED_GITHUB_TOKEN]", "credential"
    ),

    # GitHub fine-grained tokens: github_pat_ prefix
    (
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        "github_pat", "[REDACTED_GITHUB_PAT]", "credential"
    ),

    # Anthropic API keys: sk-ant- prefix
    (
        re.compile(r"(?:[A-Za-z_\-]*api[_\-]?key\s*[=:]\s*['\"]?)?sk-ant-[A-Za-z0-9\-]{20,}['\"]?", re.IGNORECASE),
        "anthropic_key", "[REDACTED_ANTHROPIC_KEY]", "credential"
    ),

    # Generic API key patterns: api_key=..., apikey=..., api-key: ...
    (
        re.compile(r"(?:api[_\-]?key|apikey|api[_\-]?secret|api[_\-]?token)\s*[=:]\s*['\"]?[A-Za-z0-9\-_]{16,}['\"]?", re.IGNORECASE),
        "generic_api_key", "[REDACTED_API_KEY]", "credential"
    ),

    # PEM private key blocks: -----BEGIN (RSA|EC|DSA|PRIVATE) ... -----END
    (
        re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE\s+KEY-----"),
        "private_key", "[REDACTED_PRIVATE_KEY]", "credential"
    ),

    # ── PII (severity: pii → risk_level: medium) ──

    # Email addresses
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "email", "[REDACTED_EMAIL]", "pii"
    ),

    # US Social Security Numbers: XXX-XX-XXXX (with dashes or spaces)
    (
        re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
        "ssn", "[REDACTED_SSN]", "pii"
    ),

    # Credit card numbers: 13-19 digits, optionally separated by dashes or spaces
    # Covers Visa (4...), Mastercard (5...), Amex (3...), etc.
    (
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"),
        "credit_card", "[REDACTED_CC]", "pii"
    ),

    # Phone numbers: various formats
    # +1-555-123-4567, (555) 123-4567, 555.123.4567, +61 2 1234 5678, etc.
    (
        re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"),
        "phone", "[REDACTED_PHONE]", "pii"
    ),
]


# ─── Scanning and Redaction Logic ────────────────────────────────────


def scan_and_redact(text: str) -> PIIScanResult:
    """
    Scan text for PII and secrets, then produce a redacted version.

    How it works:
    1. Run every pattern against the text, collect all matches
    2. Sort matches by position (start index), longest first for ties
       — this prevents a shorter match from partially redacting a longer one
    3. Walk through matches in reverse order (right to left) to redact
       — replacing right-to-left preserves earlier string positions
    4. Determine overall risk level from the highest severity found

    Args:
        text: The string to scan (agent output, document content, etc.)

    Returns:
        PIIScanResult with risk level, findings list, and redacted text.
    """
    findings: list[Finding] = []

    # Step 1: collect all matches from all patterns
    for pattern, finding_type, redaction_label, severity in PII_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding(
                finding_type=finding_type,
                matched_text=match.group(),
                span=(match.start(), match.end()),
                severity=severity,
            ))

    # Step 2: sort by start position, then by length (longest first)
    # This handles overlapping matches — the longest match wins
    findings.sort(key=lambda f: (f.span[0], -(f.span[1] - f.span[0])))

    # Remove overlapping findings — keep the first (longest) one
    deduplicated: list[Finding] = []
    last_end = -1
    for finding in findings:
        if finding.span[0] >= last_end:
            deduplicated.append(finding)
            last_end = finding.span[1]
    findings = deduplicated

    # Step 3: redact right-to-left so positions stay valid
    redacted = text
    for finding in reversed(findings):
        start, end = finding.span
        # Look up the redaction label for this finding type
        label = _get_redaction_label(finding.finding_type)
        redacted = redacted[:start] + label + redacted[end:]

    # Step 4: determine overall risk level
    has_credential = any(f.severity == "credential" for f in findings)
    has_pii = any(f.severity == "pii" for f in findings)

    if has_credential:
        risk_level = "high"
    elif has_pii:
        risk_level = "medium"
    else:
        risk_level = "low"

    return PIIScanResult(
        risk_level=risk_level,
        findings=findings,
        redacted_text=redacted,
        finding_count=len(findings),
    )


def _get_redaction_label(finding_type: str) -> str:
    """Map a finding type to its redaction label."""
    labels = {
        "email": "[REDACTED_EMAIL]",
        "phone": "[REDACTED_PHONE]",
        "ssn": "[REDACTED_SSN]",
        "credit_card": "[REDACTED_CC]",
        "aws_access_key": "[REDACTED_AWS_KEY]",
        "aws_secret_key": "[REDACTED_AWS_SECRET]",
        "github_token": "[REDACTED_GITHUB_TOKEN]",
        "github_pat": "[REDACTED_GITHUB_PAT]",
        "anthropic_key": "[REDACTED_ANTHROPIC_KEY]",
        "generic_api_key": "[REDACTED_API_KEY]",
        "private_key": "[REDACTED_PRIVATE_KEY]",
    }
    return labels.get(finding_type, "[REDACTED]")