from __future__ import annotations

import math
import re
from urllib.parse import urlparse


SUSPICIOUS_URL_TOKENS = {
    "login",
    "verify",
    "secure",
    "account",
    "bank",
    "wallet",
    "reset",
    "password",
    "mfa",
    "otp",
}

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "adf.ly",
}

SUSPICIOUS_TLDS = {
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".pw",
    ".xyz",
    ".top",
    ".online",
    ".ru",
}

EMAIL_URGENCY_TERMS = {
    "urgent",
    "immediately",
    "act now",
    "limited time",
    "suspended",
    "locked",
    "verify now",
    "final notice",
}

EMAIL_SENSITIVE_TERMS = {
    "password",
    "otp",
    "pin",
    "wire transfer",
    "gift card",
    "bank account",
    "crypto",
    "social security",
}

FILE_DANGEROUS_EXTENSIONS = {
    ".exe",
    ".dll",
    ".scr",
    ".bat",
    ".cmd",
    ".js",
    ".vbs",
    ".ps1",
    ".hta",
}

FILE_MACRO_EXTENSIONS = {
    ".docm",
    ".xlsm",
    ".pptm",
}


def _shannon_entropy(value: bytes | str) -> float:
    if not value:
        return 0.0

    if isinstance(value, str):
        data = value.encode("utf-8", errors="ignore")
    else:
        data = value

    counts = {}
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1

    length = len(data)
    entropy = 0.0
    for count in counts.values():
        prob = count / length
        if prob > 0:
            entropy -= prob * (math.log(prob) / math.log(2.0))
    return float(entropy)


def _severity_weight(severity: str) -> int:
    if severity == "critical":
        return 4
    if severity == "high":
        return 3
    if severity == "medium":
        return 2
    return 1


def _build_result(rules: list[dict]) -> dict:
    total_score = sum(_severity_weight(rule["severity"]) for rule in rules)
    high_risk = any(rule["severity"] in {"high", "critical"} for rule in rules)
    critical = any(rule["severity"] == "critical" for rule in rules)
    return {
        "matched_rules": rules,
        "score": total_score,
        "high_risk": high_risk,
        "critical": critical,
        "has_matches": bool(rules),
    }


def inspect_url_rules(url: str) -> dict:
    value = str(url or "").strip()
    if not value:
        return _build_result([])

    normalized = value if "://" in value else f"http://{value}"
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").split("@")[-1].split(":")[0].lower()
    lowered = normalized.lower()

    rules: list[dict] = []

    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", host):
        rules.append(
            {
                "id": "url_ip_host",
                "severity": "critical",
                "message": "URL host is a direct IP address.",
            }
        )

    if "@" in lowered:
        rules.append(
            {
                "id": "url_at_symbol",
                "severity": "high",
                "message": "URL contains @ symbol which can hide true host.",
            }
        )

    if any(shortener in host for shortener in SHORTENER_DOMAINS):
        if any(token in lowered for token in SUSPICIOUS_URL_TOKENS):
            rules.append(
                {
                    "id": "url_shortener_sensitive",
                    "severity": "critical",
                    "message": "Shortened URL includes credential or banking lure tokens.",
                }
            )
        else:
            rules.append(
                {
                    "id": "url_shortener",
                    "severity": "medium",
                    "message": "URL uses a shortening service and should be verified.",
                }
            )

    if "xn--" in host:
        rules.append(
            {
                "id": "url_punycode",
                "severity": "high",
                "message": "URL uses punycode and may be a homograph domain.",
            }
        )

    if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS) and any(token in lowered for token in SUSPICIOUS_URL_TOKENS):
        rules.append(
            {
                "id": "url_suspicious_tld_tokens",
                "severity": "high",
                "message": "Suspicious TLD combined with phishing-sensitive tokens.",
            }
        )

    if len(lowered) >= 180 and lowered.count("%") >= 6:
        rules.append(
            {
                "id": "url_obfuscated_encoding",
                "severity": "high",
                "message": "Long URL with heavy percent encoding suggests obfuscation.",
            }
        )

    if lowered.count("//") > 2:
        rules.append(
            {
                "id": "url_redirect_chain",
                "severity": "medium",
                "message": "URL contains multiple redirect markers.",
            }
        )

    return _build_result(rules)


def _inspect_text_rules(text: str, channel: str) -> dict:
    value = str(text or "").lower()
    rules: list[dict] = []

    has_link = bool(re.search(r"https?://|www\.", value))
    urgency_hits = [term for term in EMAIL_URGENCY_TERMS if term in value]
    sensitive_hits = [term for term in EMAIL_SENSITIVE_TERMS if term in value]

    if has_link and urgency_hits and sensitive_hits:
        rules.append(
            {
                "id": f"{channel}_link_urgency_sensitive",
                "severity": "critical",
                "message": "Message combines urgency, sensitive requests, and embedded links.",
            }
        )
    elif has_link and (urgency_hits or sensitive_hits):
        rules.append(
            {
                "id": f"{channel}_suspicious_link_combo",
                "severity": "high",
                "message": "Message has suspicious link context and social engineering indicators.",
            }
        )

    if value.count("!") >= 3 and urgency_hits:
        rules.append(
            {
                "id": f"{channel}_high_pressure_punctuation",
                "severity": "medium",
                "message": "Excessive urgency and punctuation pressure detected.",
            }
        )

    if re.search(r"\b(?:otp|code|password|pin)\b.{0,24}\b(?:share|send|confirm|verify)\b", value):
        rules.append(
            {
                "id": f"{channel}_credential_harvest",
                "severity": "high",
                "message": "Potential credential harvesting phrasing detected.",
            }
        )

    return _build_result(rules)


def inspect_email_rules(subject: str, message: str) -> dict:
    composed = f"{subject or ''}\n{message or ''}"
    return _inspect_text_rules(composed, "email")


def inspect_message_rules(message: str) -> dict:
    return _inspect_text_rules(message, "message")


def inspect_file_rules(file_bytes: bytes, filename: str) -> dict:
    name = str(filename or "").strip().lower()
    rules: list[dict] = []

    if re.search(r"\.(pdf|doc|xls|jpg|png)\.(exe|scr|js|vbs|bat|cmd|ps1)$", name):
        rules.append(
            {
                "id": "file_double_extension",
                "severity": "critical",
                "message": "File uses deceptive double extension.",
            }
        )

    for extension in FILE_DANGEROUS_EXTENSIONS:
        if name.endswith(extension):
            rules.append(
                {
                    "id": "file_dangerous_extension",
                    "severity": "high",
                    "message": f"Executable scripting extension detected ({extension}).",
                }
            )
            break

    for extension in FILE_MACRO_EXTENSIONS:
        if name.endswith(extension):
            rules.append(
                {
                    "id": "file_macro_extension",
                    "severity": "high",
                    "message": f"Macro-enabled office extension detected ({extension}).",
                }
            )
            break

    sample = file_bytes[:4096] if file_bytes else b""
    if sample:
        entropy = _shannon_entropy(sample)
        if entropy >= 7.3:
            rules.append(
                {
                    "id": "file_high_entropy",
                    "severity": "medium",
                    "message": "High entropy in file header may indicate packing/obfuscation.",
                }
            )
        if len(file_bytes) > 0 and len(file_bytes) < 4096 and entropy > 6.7:
            rules.append(
                {
                    "id": "file_tiny_obfuscated_payload",
                    "severity": "high",
                    "message": "Small file with high entropy resembles dropper payload patterns.",
                }
            )

    return _build_result(rules)


def inspect_fraud_rules(transaction: dict) -> dict:
    payload = transaction or {}
    rules: list[dict] = []

    amount = max(float(payload.get("amount", 0.0) or 0.0), 0.0)
    tx_type = str(payload.get("type", "TRANSFER")).upper()
    step = int(payload.get("step", 1) or 1)
    hour = (step - 1) % 24
    is_night = hour >= 22 or hour <= 4

    if tx_type in {"TRANSFER", "CASH_OUT"} and amount >= 250000:
        rules.append(
            {
                "id": "fraud_high_value_transfer",
                "severity": "high",
                "message": "High-value transfer/cash-out transaction.",
            }
        )

    if is_night and amount >= 100000:
        rules.append(
            {
                "id": "fraud_night_high_value",
                "severity": "high",
                "message": "High-value transaction during low-activity window.",
            }
        )

    orig_tx_count = float(payload.get("orig_tx_count_last_24h", 1.0) or 1.0)
    dest_in_degree = float(payload.get("dest_in_degree", 1.0) or 1.0)

    if orig_tx_count >= 25:
        rules.append(
            {
                "id": "fraud_velocity_spike",
                "severity": "high",
                "message": "Transaction velocity spike for origin account.",
            }
        )

    if amount >= 500000 and dest_in_degree <= 1:
        rules.append(
            {
                "id": "fraud_new_destination_high_value",
                "severity": "critical",
                "message": "Large transfer to destination with low historical connectivity.",
            }
        )

    return _build_result(rules)
