from __future__ import annotations

from dataclasses import dataclass


MALICIOUS_VERDICTS = {"SPAM", "SCAM", "MALICIOUS", "MALWARE", "FRAUD"}


@dataclass(slots=True)
class ConfidenceThresholds:
    allow_confidence: float = 0.90
    uncertain_confidence: float = 0.70
    block_malicious_confidence: float = 0.95


def decide_scan_action(
    verdict: str,
    confidence: float,
    rules: dict,
    anomaly: dict,
    thresholds: ConfidenceThresholds,
) -> dict:
    verdict_upper = str(verdict or "").upper()
    is_malicious = verdict_upper in MALICIOUS_VERDICTS
    is_anomalous = bool((anomaly or {}).get("flagged", False))
    has_high_risk_rule = bool((rules or {}).get("high_risk", False))
    has_critical_rule = bool((rules or {}).get("critical", False))

    action = "allow"
    stage = "high_confidence"
    route = "edge"
    reasons: list[str] = []

    if is_malicious and confidence >= thresholds.block_malicious_confidence:
        action = "block"
        stage = "high_confidence"
        route = "edge"
        reasons.append("high-confidence malicious prediction")
    elif confidence < thresholds.uncertain_confidence:
        action = "quarantine"
        stage = "uncertain"
        route = "cloud"
        reasons.append("low model confidence")
    elif confidence < thresholds.allow_confidence:
        action = "allow_with_monitoring"
        stage = "gray_zone"
        route = "active_learning"
        reasons.append("mid-confidence prediction")

    if has_critical_rule:
        action = "block" if confidence >= thresholds.uncertain_confidence else "quarantine"
        stage = "heuristic_override"
        route = "cloud"
        reasons.append("critical heuristic rule triggered")
    elif has_high_risk_rule and not is_malicious:
        action = "quarantine"
        stage = "heuristic_override"
        route = "cloud"
        reasons.append("high-risk heuristic rule triggered")

    if is_anomalous and not is_malicious:
        action = "escalate"
        stage = "anomaly_override"
        route = "cloud"
        reasons.append("anomaly detector conflict with benign verdict")

    send_to_cloud = route == "cloud" or action in {"quarantine", "escalate"}
    requires_human_review = stage in {"uncertain", "gray_zone", "anomaly_override"} or has_high_risk_rule
    log_for_labeling = stage in {"gray_zone", "uncertain", "anomaly_override"} or is_anomalous
    quarantine_temp = action in {"quarantine", "escalate"}

    return {
        "action": action,
        "stage": stage,
        "route": route,
        "reasons": reasons or ["policy-default"],
        "send_to_cloud": send_to_cloud,
        "requires_human_review": requires_human_review,
        "log_for_labeling": log_for_labeling,
        "quarantine_temp": quarantine_temp,
        "is_malicious_label": is_malicious,
    }
