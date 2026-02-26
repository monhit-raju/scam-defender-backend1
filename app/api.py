from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import io
import json
import logging
import math
import mimetypes
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request, send_file
from flask_jwt_extended import (
    create_access_token,
    decode_token,
    get_jwt_identity,
    jwt_required,
)
from werkzeug.security import check_password_hash, generate_password_hash

from app.services.gating_policy import decide_scan_action
from app.services.model_service import ModelServiceError
from app.services.mongo_store import ScanPersistenceInput, parse_object_id
from app.services.risk_rules import (
    inspect_email_rules,
    inspect_file_rules,
    inspect_fraud_rules,
    inspect_message_rules,
    inspect_url_rules,
)

api_bp = Blueprint("api", __name__)
LOGGER = logging.getLogger(__name__)

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}

SANDBOX_DANGEROUS_EXTENSIONS = {
    ".exe",
    ".dll",
    ".scr",
    ".bat",
    ".cmd",
    ".js",
    ".jse",
    ".vbs",
    ".vbe",
    ".ps1",
    ".hta",
    ".jar",
    ".com",
}

SANDBOX_MACRO_EXTENSIONS = {
    ".docm",
    ".xlsm",
    ".pptm",
}

SANDBOX_ARCHIVE_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}


def _rate_limit(bucket: str, max_requests: int, window_seconds: int = 60) -> bool:
    now = time.time()
    key = f"{bucket}:{request.remote_addr or 'unknown'}"
    with _RATE_LIMIT_LOCK:
        entries = _RATE_LIMIT_BUCKETS.setdefault(key, [])
        threshold = now - float(window_seconds)
        entries[:] = [value for value in entries if value >= threshold]
        if len(entries) >= int(max_requests):
            return False
        entries.append(now)
    return True


def _integration_token_valid(*expected_tokens: str | None) -> bool:
    provided = (
        request.headers.get("X-Integration-Token")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        or request.args.get("token", "").strip()
    )

    configured = [str(token or "").strip() for token in expected_tokens if str(token or "").strip()]
    if not configured:
        configured = [str(current_app.config.get("INTEGRATION_SHARED_TOKEN", "")).strip()]
        configured = [token for token in configured if token]

    if not configured:
        # Development fallback if no token configured.
        return True

    return bool(provided) and any(hmac.compare_digest(provided, token) for token in configured)


def _require_integration_token(*expected_tokens: str | None):
    if _integration_token_valid(*expected_tokens):
        return None
    return jsonify({"error": "invalid or missing integration token"}), 401


def _enforce_rate_limit(bucket: str, upload: bool = False):
    limit = int(
        current_app.config["API_UPLOAD_RATE_LIMIT_PER_MINUTE"]
        if upload
        else current_app.config["API_RATE_LIMIT_PER_MINUTE"]
    )
    if _rate_limit(bucket, max_requests=limit, window_seconds=60):
        return None
    return jsonify({"error": "rate limit exceeded"}), 429


def _verify_hub_signature(payload: bytes, secret: str, signature_header: str) -> bool:
    if not secret:
        return True
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.strip())


def _decode_base64_payload(value: str | None) -> bytes:
    if not value:
        return b""
    raw = str(value).strip()
    if "," in raw and "base64" in raw.split(",", 1)[0]:
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=False)
    except Exception:
        return b""


def _extract_gmail_message_text(payload: dict[str, Any]) -> str:
    parts_queue = [payload.get("payload", {})]
    collected: list[str] = []

    while parts_queue:
        part = parts_queue.pop(0) or {}
        body = part.get("body", {}) or {}
        data = body.get("data")
        mime = str(part.get("mimeType", "")).lower()
        if data and mime.startswith("text/"):
            decoded = _decode_base64_payload(data)
            if decoded:
                collected.append(decoded.decode("utf-8", errors="ignore"))
        for nested in part.get("parts", []) or []:
            parts_queue.append(nested)

    snippet = str(payload.get("snippet", "")).strip()
    combined = "\n".join(text for text in collected if text).strip()
    if combined:
        return combined
    return snippet


def _extract_gmail_headers(payload: dict[str, Any]) -> dict[str, str]:
    headers = {}
    for header in payload.get("payload", {}).get("headers", []) or []:
        key = str(header.get("name", "")).lower()
        value = str(header.get("value", ""))
        if key:
            headers[key] = value
    return headers


def _fetch_whatsapp_media(media_id: str) -> tuple[bytes, str, str]:
    token = str(current_app.config.get("WHATSAPP_ACCESS_TOKEN", "")).strip()
    if not token or not media_id:
        return b"", "", "missing_credentials"

    try:
        import requests
    except Exception:
        return b"", "", "requests_unavailable"

    headers = {"Authorization": f"Bearer {token}"}
    info_resp = requests.get(
        f"https://graph.facebook.com/v21.0/{media_id}",
        headers=headers,
        timeout=12,
    )
    if info_resp.status_code >= 400:
        return b"", "", f"metadata_error_{info_resp.status_code}"

    media_info = info_resp.json() if info_resp.content else {}
    media_url = str(media_info.get("url", ""))
    media_name = str(media_info.get("filename", "") or f"{media_id}.bin")
    if not media_url:
        return b"", media_name, "missing_media_url"

    media_resp = requests.get(media_url, headers=headers, timeout=20)
    if media_resp.status_code >= 400:
        return b"", media_name, f"download_error_{media_resp.status_code}"
    return media_resp.content, media_name, "ok"


def _collect_whatsapp_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            contacts = value.get("contacts", []) or []
            contact_name = ""
            if contacts:
                contact_name = str((contacts[0].get("profile", {}) or {}).get("name", ""))

            for message in value.get("messages", []) or []:
                msg_type = str(message.get("type", "text"))
                msg_text = ""
                attachment_info = None

                if msg_type == "text":
                    msg_text = str((message.get("text", {}) or {}).get("body", ""))
                elif msg_type in {"image", "document", "audio", "video", "sticker"}:
                    media = message.get(msg_type, {}) or {}
                    caption = str(media.get("caption", "")).strip()
                    msg_text = caption
                    attachment_info = {
                        "media_id": str(media.get("id", "")),
                        "filename": str(media.get("filename", "") or f"{msg_type}_{message.get('id', 'item')}"),
                        "mime_type": str(media.get("mime_type", "")),
                        "type": msg_type,
                    }
                elif msg_type == "interactive":
                    interactive = message.get("interactive", {}) or {}
                    title = str((interactive.get("button_reply", {}) or {}).get("title", ""))
                    msg_text = title or json.dumps(interactive)
                else:
                    msg_text = str(message.get(msg_type, "") or "")

                collected.append(
                    {
                        "message_id": str(message.get("id", "")),
                        "from": str(message.get("from", "")),
                        "timestamp": str(message.get("timestamp", "")),
                        "type": msg_type,
                        "text": msg_text,
                        "contact_name": contact_name,
                        "attachment": attachment_info,
                    }
                )
    return collected

def _get_model_service():
    return current_app.extensions["model_service"]


def _get_anomaly_service():
    return current_app.extensions["anomaly_service"]


def _get_store():
    return current_app.extensions["store"]


def _get_cloud_verifier():
    return current_app.extensions["cloud_verifier"]


def _get_thresholds():
    return current_app.extensions["confidence_thresholds"]


def _current_user(optional: bool = False):
    identity = get_jwt_identity()
    if identity is None:
        return None if optional else None
    return _get_store().compose_user(identity)


def _severity_from_risk(risk_score: float, action: str = "allow") -> str:
    score = float(risk_score or 0.0)
    if action in {"block", "quarantine", "escalate"} and score < 65.0:
        return "high"
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _excerpt(value: str, max_len: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _scan_response(record: dict) -> dict:
    return {
        "id": record["id"],
        "scan_type": record.get("scan_type", ""),
        "verdict": record.get("verdict", "UNKNOWN"),
        "severity": record.get("severity", "low"),
        "confidence": round(float(record.get("confidence", 0.0)), 4),
        "risk_score": round(float(record.get("risk_score", 0.0)), 2),
        "details": record.get("details", {}),
        "gating": record.get("gating", {}),
        "input_excerpt": record.get("input_excerpt", ""),
        "created_at": record.get("created_at"),
    }


def _serialize_alert(alert: dict) -> dict:
    return {
        "id": alert["id"],
        "scan_record_id": alert.get("scan_record_id"),
        "severity": alert.get("severity", "medium"),
        "title": alert.get("title", ""),
        "message": alert.get("message", ""),
        "acknowledged": bool(alert.get("acknowledged", False)),
        "created_at": alert.get("created_at"),
    }


def _sanitize_sample_payload(scan_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    max_inline_bytes = int(current_app.config.get("CLOUD_MAX_INLINE_FILE_BYTES", 5 * 1024 * 1024))

    if scan_type == "email":
        sanitized_attachments = []
        for attachment in payload.get("attachments", []):
            file_bytes = attachment.get("file_bytes", b"")
            filename = str(attachment.get("filename", "attachment"))
            include_content = len(file_bytes) <= max_inline_bytes
            item = {
                "filename": filename,
                "size_bytes": len(file_bytes),
                "sha256": hashlib.sha256(file_bytes).hexdigest() if file_bytes else "",
                "has_inline_content": include_content,
            }
            if include_content and file_bytes:
                item["file_content_base64"] = base64.b64encode(file_bytes).decode("ascii")
            sanitized_attachments.append(item)

        return {
            "subject": payload.get("subject", ""),
            "message": payload.get("message", ""),
            "attachments": sanitized_attachments,
            "attachment_sandbox": payload.get("attachment_sandbox", []),
        }

    if scan_type != "file":
        return payload

    file_bytes = payload.get("file_bytes", b"")
    filename = str(payload.get("filename", "uploaded_file"))
    sha256_hash = hashlib.sha256(file_bytes).hexdigest() if file_bytes else ""

    include_content = len(file_bytes) <= max_inline_bytes

    sanitized = {
        "filename": filename,
        "size_bytes": len(file_bytes),
        "sha256": sha256_hash,
        "has_inline_content": include_content,
        "attachment_sandbox": payload.get("attachment_sandbox", []),
    }
    if include_content and file_bytes:
        sanitized["file_content_base64"] = base64.b64encode(file_bytes).decode("ascii")
    return sanitized


def _rule_severity_weight(severity: str) -> int:
    value = str(severity or "").lower()
    if value == "critical":
        return 4
    if value == "high":
        return 3
    if value == "medium":
        return 2
    return 1


def _merge_rule_results(primary: dict, secondary: dict) -> dict:
    left = primary or {}
    right = secondary or {}
    merged_rules = [*(left.get("matched_rules", []) or []), *(right.get("matched_rules", []) or [])]
    return {
        "matched_rules": merged_rules,
        "score": sum(_rule_severity_weight(rule.get("severity", "")) for rule in merged_rules),
        "high_risk": any(rule.get("severity") in {"high", "critical"} for rule in merged_rules),
        "critical": any(rule.get("severity") == "critical" for rule in merged_rules),
        "has_matches": bool(merged_rules),
    }


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts: dict[int, int] = {}
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log(prob, 2)
    return float(entropy)


def _sandbox_analyze_attachment(file_bytes: bytes, filename: str, source: str = "file_upload") -> dict[str, Any]:
    raw = file_bytes or b""
    normalized_name = str(filename or "uploaded_file")
    lowered_name = normalized_name.lower()
    suffix = Path(lowered_name).suffix
    guessed_type = mimetypes.guess_type(normalized_name)[0] or "application/octet-stream"

    signals: list[dict[str, Any]] = []
    score = 0

    def add_signal(signal_id: str, severity: str, message: str):
        nonlocal score
        weight = _rule_severity_weight(severity)
        score += weight
        signals.append(
            {
                "id": signal_id,
                "severity": severity,
                "message": message,
            }
        )

    if re.search(r"\.(pdf|doc|xls|jpg|png|txt)\.(exe|scr|js|vbs|bat|cmd|ps1)$", lowered_name):
        add_signal("sandbox_double_extension", "critical", "Attachment uses deceptive double extension.")

    if suffix in SANDBOX_DANGEROUS_EXTENSIONS:
        add_signal("sandbox_dangerous_extension", "high", f"Dangerous executable/script extension detected ({suffix}).")

    if suffix in SANDBOX_MACRO_EXTENSIONS:
        add_signal("sandbox_macro_extension", "high", f"Macro-enabled office extension detected ({suffix}).")

    if suffix in SANDBOX_ARCHIVE_EXTENSIONS:
        add_signal("sandbox_archive_attachment", "medium", "Archive attachment requires nested file review.")

    if raw[:2] == b"MZ" and suffix not in {".exe", ".dll", ".sys", ".com", ".scr"}:
        add_signal("sandbox_mz_header_mismatch", "high", "Executable header detected with non-executable extension.")

    sample = raw[:4096]
    entropy = _byte_entropy(sample)
    if entropy >= 7.25:
        add_signal("sandbox_high_entropy", "medium", "High entropy suggests packing or obfuscation.")

    if len(raw) <= 4096 and entropy >= 6.8:
        add_signal("sandbox_tiny_obfuscated_payload", "high", "Very small file with high entropy resembles dropper patterns.")

    lowered_sample = sample.lower()
    if b"powershell" in lowered_sample or b"cmd.exe" in lowered_sample or b"wscript" in lowered_sample:
        add_signal("sandbox_script_launcher_text", "high", "Script launcher indicators found in file content.")

    if suffix == ".pdf" and (b"/javascript" in lowered_sample or b"/js" in lowered_sample):
        add_signal("sandbox_pdf_script", "high", "PDF contains embedded JavaScript indicators.")

    if raw.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
                names = [name.lower() for name in archive.namelist()]
                if any("vba" in name for name in names):
                    add_signal("sandbox_macro_payload", "high", "Archive contains VBA macro payload markers.")
                if any(Path(name).suffix in SANDBOX_DANGEROUS_EXTENSIONS for name in names):
                    add_signal("sandbox_dangerous_nested_file", "critical", "Archive contains executable/script attachment.")
        except Exception:
            add_signal("sandbox_archive_parse_error", "medium", "Archive structure could not be parsed for safe inspection.")

    suspicious = any(signal["severity"] in {"high", "critical"} for signal in signals)

    return {
        "source": source,
        "filename": normalized_name,
        "extension": suffix,
        "content_type": guessed_type,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
        "header_entropy": round(float(entropy), 4),
        "score": score,
        "suspicious": suspicious,
        "signals": signals,
    }


def _sandbox_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    values = artifacts or []
    suspicious_count = sum(1 for item in values if item.get("suspicious"))
    max_score = max((int(item.get("score", 0)) for item in values), default=0)
    return {
        "total_files": len(values),
        "suspicious_count": suspicious_count,
        "max_score": max_score,
        "has_suspicious_files": suspicious_count > 0,
    }


def _sandbox_rule_result(scan_type: str, artifacts: list[dict[str, Any]]) -> dict:
    summary = _sandbox_summary(artifacts)
    if not summary["has_suspicious_files"]:
        return {"matched_rules": [], "score": 0, "high_risk": False, "critical": False, "has_matches": False}

    critical_hits = sum(
        1
        for artifact in artifacts
        for signal in artifact.get("signals", [])
        if signal.get("severity") == "critical"
    )
    severity = "critical" if critical_hits > 0 else "high"
    scope = "attachment" if scan_type in {"email", "message"} else "file"
    return {
        "matched_rules": [
            {
                "id": f"{scan_type}_sandbox_suspicious_{scope}",
                "severity": severity,
                "message": f"Sandbox detected {summary['suspicious_count']} suspicious {scope}(s).",
            }
        ],
        "score": _rule_severity_weight(severity),
        "high_risk": True,
        "critical": severity == "critical",
        "has_matches": True,
    }


def _apply_sandbox_to_result(model_result: dict[str, Any], scan_type: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(model_result or {})
    details = dict(result.get("details", {}) or {})

    summary = _sandbox_summary(artifacts)
    details["sandbox"] = {
        "summary": summary,
        "artifacts": artifacts,
    }

    if summary["has_suspicious_files"]:
        base_risk = float(result.get("risk_score", 0.0))
        boost = min(35.0, (summary["suspicious_count"] * 8.0) + (summary["max_score"] * 1.6))
        updated_risk = min(99.9, base_risk + boost)
        result["risk_score"] = round(updated_risk, 2)
        details["sandbox"]["risk_adjustment"] = {
            "base_risk_score": round(base_risk, 2),
            "boost": round(boost, 2),
            "final_risk_score": round(updated_risk, 2),
        }

        current_verdict = str(result.get("verdict", "")).upper()
        if scan_type == "email" and current_verdict == "HAM" and updated_risk >= 45.0:
            result["verdict"] = "SPAM"
        if scan_type == "file" and current_verdict in {"GOODWARE", "NOT_PE_OR_INVALID"} and updated_risk >= 55.0:
            result["verdict"] = "MALWARE"

        verdict_after = str(result.get("verdict", "")).upper()
        current_confidence = float(result.get("confidence", 0.0))
        if verdict_after in {"SPAM", "SCAM", "MALICIOUS", "MALWARE", "FRAUD"}:
            result["confidence"] = round(max(current_confidence, updated_risk / 100.0), 4)
        else:
            result["confidence"] = round(max(current_confidence, 1.0 - (updated_risk / 100.0)), 4)

    result["details"] = details
    return result


def _evaluate_rules(scan_type: str, payload: dict[str, Any]) -> dict:
    if scan_type == "email":
        base = inspect_email_rules(payload.get("subject", ""), payload.get("message", ""))
        sandbox = _sandbox_rule_result("email", payload.get("attachment_sandbox", []))
        return _merge_rule_results(base, sandbox)
    if scan_type == "message":
        base = inspect_message_rules(payload.get("message", ""))
        sandbox = _sandbox_rule_result("message", payload.get("attachment_sandbox", []))
        return _merge_rule_results(base, sandbox)
    if scan_type == "url":
        return inspect_url_rules(payload.get("url", ""))
    if scan_type == "file":
        base = inspect_file_rules(payload.get("file_bytes", b""), payload.get("filename", ""))
        sandbox = _sandbox_rule_result("file", payload.get("attachment_sandbox", []))
        return _merge_rule_results(base, sandbox)
    if scan_type == "fraud":
        return inspect_fraud_rules(payload.get("transaction", {}))
    return {"matched_rules": [], "score": 0, "high_risk": False, "critical": False, "has_matches": False}


def _evaluate_anomaly(scan_type: str, payload: dict[str, Any], model_details: dict[str, Any]) -> dict:
    service = _get_anomaly_service()
    if scan_type == "email":
        return service.assess_email(payload.get("subject", ""), payload.get("message", ""))
    if scan_type == "message":
        return service.assess_message(payload.get("message", ""))
    if scan_type == "url":
        return service.assess_url(payload.get("url", ""))
    if scan_type == "file":
        return service.assess_file(
            payload.get("file_bytes", b""),
            filename=payload.get("filename", ""),
            pe_features=model_details.get("pe_aggregate_features"),
        )
    if scan_type == "fraud":
        return service.assess_fraud(model_details)
    return {"flagged": False, "score": None, "source": "not_applicable"}


def _maybe_escalate(
    user_id: str | None,
    scan_type: str,
    sample_payload: dict[str, Any],
    result: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    details = result.get("details", {})
    store = _get_store()

    if policy.get("log_for_labeling"):
        queue_item = store.enqueue_active_learning(
            user_id=user_id,
            scan_type=scan_type,
            sample_payload=sample_payload,
            model_output=result,
            reason="uncertain_or_anomalous_prediction",
            priority=80 if policy.get("stage") == "uncertain" else 55,
            status="pending",
        )
        details["active_learning_item_id"] = queue_item["id"]

    if policy.get("send_to_cloud"):
        cloud_queue_item = store.enqueue_cloud_verification(
            user_id=user_id,
            scan_type=scan_type,
            sample_payload=sample_payload,
            reason=", ".join(policy.get("reasons", [])),
            status="queued",
        )
        cloud_result = _get_cloud_verifier().submit(
            scan_type=scan_type,
            payload=sample_payload,
            model_output=result,
            policy=policy,
        )
        final_status = "verified" if cloud_result.get("status") == "ok" else "failed"
        store.update_cloud_verification(
            cloud_queue_item["id"],
            status=final_status,
            cloud_response=cloud_result,
        )
        details["cloud_verification"] = {
            "queue_id": cloud_queue_item["id"],
            **cloud_result,
        }

    if policy.get("requires_human_review"):
        details["review_recommended"] = True

    result["details"] = details
    return result


def _persist_scan(user_id: str | None, scan_type: str, input_excerpt: str, result: dict[str, Any], policy: dict[str, Any]) -> dict:
    store = _get_store()
    severity = _severity_from_risk(result.get("risk_score", 0.0), action=policy.get("action", "allow"))

    record = store.create_scan_record(
        ScanPersistenceInput(
            user_id=parse_object_id(user_id),
            scan_type=scan_type,
            input_excerpt=_excerpt(input_excerpt),
            verdict=str(result.get("verdict", "UNKNOWN")),
            severity=severity,
            confidence=float(result.get("confidence", 0.0)),
            risk_score=float(result.get("risk_score", 0.0)),
            details=result.get("details", {}),
            gating=policy,
        )
    )

    if severity in {"high", "critical"} or policy.get("action") in {"block", "quarantine", "escalate"}:
        store.create_alert(
            scan_record_id=record["id"],
            user_id=user_id,
            severity=severity if severity in {"high", "critical"} else "high",
            title=f"{scan_type.upper()} Threat Escalation",
            message=(
                f"{scan_type.upper()} scan routed action '{policy.get('action')}' "
                f"for verdict {result.get('verdict')} at risk {float(result.get('risk_score', 0.0)):.2f}%."
            ),
        )

    return record


def _build_scan_result(scan_type: str, payload: dict[str, Any], model_result: dict[str, Any]) -> tuple[dict, dict]:
    thresholds = _get_thresholds()
    rules = _evaluate_rules(scan_type, payload)
    anomaly = _evaluate_anomaly(scan_type, payload, model_result.get("details", {}))

    policy = decide_scan_action(
        verdict=model_result.get("verdict", ""),
        confidence=float(model_result.get("confidence", 0.0)),
        rules=rules,
        anomaly=anomaly,
        thresholds=thresholds,
    )

    details = model_result.get("details", {}) or {}
    details["heuristics"] = rules
    details["anomaly"] = anomaly
    details["policy_snapshot"] = {
        "allow_confidence": thresholds.allow_confidence,
        "uncertain_confidence": thresholds.uncertain_confidence,
        "block_malicious_confidence": thresholds.block_malicious_confidence,
    }
    model_result["details"] = details
    return model_result, policy


def _persist_model_scan(
    identity: str | None,
    scan_type: str,
    payload: dict[str, Any],
    model_result: dict[str, Any],
    excerpt: str,
) -> dict[str, Any]:
    model_result, policy = _build_scan_result(scan_type, payload, model_result)
    sanitized_payload = _sanitize_sample_payload(scan_type, payload) if scan_type in {"email", "file"} else payload
    model_result = _maybe_escalate(identity, scan_type, sanitized_payload, model_result, policy)
    return _persist_scan(identity, scan_type, excerpt, model_result, policy)


@api_bp.errorhandler(ModelServiceError)
def handle_model_error(exc: ModelServiceError):
    return jsonify({"error": str(exc)}), 400


@api_bp.route("/health", methods=["GET"])
def health_check():
    store = _get_store()
    cloud = _get_cloud_verifier()
    model_status = _get_model_service().get_status()
    thresholds = _get_thresholds()

    return jsonify(
        {
            "status": "ok",
            "timestamp": dt.datetime.utcnow().isoformat(),
            "models": model_status,
            "database": {
                "engine": "mongodb",
                "connected": store.ping(),
                "db_name": current_app.config["MONGO_DB_NAME"],
            },
            "cloud_verification": {
                "enabled": cloud.enabled,
                "endpoint": cloud.endpoint_url if cloud.enabled else "",
            },
            "policy": {
                "allow_confidence": thresholds.allow_confidence,
                "uncertain_confidence": thresholds.uncertain_confidence,
                "block_malicious_confidence": thresholds.block_malicious_confidence,
            },
        }
    )


@api_bp.route("/auth/register", methods=["POST"])
def register():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    full_name = str(payload.get("full_name", "")).strip() or "Scam Defender User"

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400

    try:
        user = _get_store().create_user(
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409

    token = create_access_token(identity=user["id"])
    user_payload = _get_store().compose_user(user["id"])
    return jsonify({"access_token": token, "user": user_payload}), 201


@api_bp.route("/auth/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    user = _get_store().get_user_by_email(email)
    if user is None or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "invalid credentials"}), 401

    token = create_access_token(identity=user["id"])
    user_payload = _get_store().compose_user(user["id"])
    return jsonify({"access_token": token, "user": user_payload})


@api_bp.route("/auth/me", methods=["GET"])
@jwt_required()
def me():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"user": user})


@api_bp.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"profile": user.get("profile", {}), "full_name": user.get("full_name", "")})


@api_bp.route("/profile", methods=["PUT"])
@jwt_required()
def update_profile():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    payload = request.get_json(silent=True) or {}
    updated_user = _get_store().update_profile(user["id"], payload)
    return jsonify({"user": updated_user})


@api_bp.route("/settings/privacy", methods=["GET"])
@jwt_required()
def get_privacy_settings():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    settings = _get_store().get_privacy(user["id"])
    return jsonify(
        {
            "privacy": {
                "two_factor_enabled": settings.get("two_factor_enabled", False),
                "email_alerts": settings.get("email_alerts", True),
                "sms_alerts": settings.get("sms_alerts", False),
                "share_anonymized_analytics": settings.get("share_anonymized_analytics", True),
                "data_retention_days": settings.get("data_retention_days", 90),
                "profile_visibility": settings.get("profile_visibility", "private"),
            }
        }
    )


@api_bp.route("/settings/privacy", methods=["PUT"])
@jwt_required()
def update_privacy_settings():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    payload = request.get_json(silent=True) or {}
    updated = _get_store().update_privacy(user["id"], payload)
    return jsonify(
        {
            "privacy": {
                "two_factor_enabled": updated.get("two_factor_enabled", False),
                "email_alerts": updated.get("email_alerts", True),
                "sms_alerts": updated.get("sms_alerts", False),
                "share_anonymized_analytics": updated.get("share_anonymized_analytics", True),
                "data_retention_days": updated.get("data_retention_days", 90),
                "profile_visibility": updated.get("profile_visibility", "private"),
            }
        }
    )


@api_bp.route("/feedback", methods=["POST"])
@jwt_required(optional=True)
def submit_feedback():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "feedback message is required"}), 400

    identity = get_jwt_identity()
    entry = _get_store().insert_feedback(identity, payload)
    return jsonify({"message": "feedback submitted", "feedback_id": entry["id"]}), 201


@api_bp.route("/feedback", methods=["GET"])
@jwt_required()
def list_feedback():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    entries = _get_store().list_feedback(user["id"], limit=100)
    return jsonify({"items": entries})


def _integration_scan_email(
    identity: str | None,
    subject: str,
    message: str,
    attachments: list[dict[str, Any]] | None = None,
    source: str = "integration_ingest",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_attachment_bytes = int(current_app.config.get("CLOUD_MAX_INLINE_FILE_BYTES", 5 * 1024 * 1024)) * 2
    attachments = [
        item
        for item in (attachments or [])
        if len(item.get("file_bytes", b"")) <= max_attachment_bytes
    ]
    metadata = metadata or {}

    attachment_sandbox = [
        _sandbox_analyze_attachment(
            attachment.get("file_bytes", b""),
            attachment.get("filename", "attachment"),
            source=source,
        )
        for attachment in attachments
    ]

    if not subject.strip() and not message.strip() and attachments:
        subject = "[Attachment-only email]"
        message = "No email body provided. Risk score includes attachment sandbox signals."

    model_result = _get_model_service().predict_email(subject, message)
    model_result = _apply_sandbox_to_result(model_result, "email", attachment_sandbox)
    details = dict(model_result.get("details", {}) or {})
    details["integration"] = {"source": source, **metadata}
    model_result["details"] = details

    scan_payload = {
        "subject": subject,
        "message": message,
        "attachments": attachments,
        "attachment_sandbox": attachment_sandbox,
    }
    record = _persist_model_scan(identity, "email", scan_payload, model_result, f"{subject}\n{message}")
    return _scan_response(record)


def _integration_scan_message(
    identity: str | None,
    message_text: str,
    source: str = "integration_ingest",
    metadata: dict[str, Any] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    max_attachment_bytes = int(current_app.config.get("CLOUD_MAX_INLINE_FILE_BYTES", 5 * 1024 * 1024)) * 2
    attachments = [
        item
        for item in (attachments or [])
        if len(item.get("file_bytes", b"")) <= max_attachment_bytes
    ]
    attachment_sandbox = [
        _sandbox_analyze_attachment(
            attachment.get("file_bytes", b""),
            attachment.get("filename", "attachment"),
            source=source,
        )
        for attachment in attachments
    ]

    scan_text = str(message_text or "").strip() or "[Attachment-only message]"
    model_result = _get_model_service().predict_message(scan_text)
    if attachment_sandbox:
        model_result = _apply_sandbox_to_result(model_result, "message", attachment_sandbox)

    details = dict(model_result.get("details", {}) or {})
    details["integration"] = {"source": source, **metadata}
    if attachment_sandbox:
        details.setdefault("sandbox", {}).setdefault("artifacts", attachment_sandbox)
    model_result["details"] = details

    scan_payload = {
        "message": scan_text,
        "attachment_sandbox": attachment_sandbox,
    }
    record = _persist_model_scan(identity, "message", scan_payload, model_result, scan_text)
    return _scan_response(record)


def _get_google_access_token() -> str:
    client_id = str(current_app.config.get("GMAIL_CLIENT_ID", "")).strip()
    client_secret = str(current_app.config.get("GMAIL_CLIENT_SECRET", "")).strip()
    refresh_token = str(current_app.config.get("GMAIL_REFRESH_TOKEN", "")).strip()
    if not all([client_id, client_secret, refresh_token]):
        raise ModelServiceError("Missing Gmail OAuth credentials in environment variables.")

    try:
        import requests
    except Exception as exc:
        raise ModelServiceError("requests package is required for Gmail integration.") from exc

    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    if response.status_code >= 400:
        raise ModelServiceError(f"Gmail OAuth token refresh failed ({response.status_code}).")
    payload = response.json() if response.content else {}
    access_token = str(payload.get("access_token", "")).strip()
    if not access_token:
        raise ModelServiceError("Gmail OAuth response did not include access_token.")
    return access_token


def _get_microsoft_access_token() -> str:
    tenant_id = str(current_app.config.get("MICROSOFT_TENANT_ID", "")).strip()
    client_id = str(current_app.config.get("MICROSOFT_CLIENT_ID", "")).strip()
    client_secret = str(current_app.config.get("MICROSOFT_CLIENT_SECRET", "")).strip()
    if not all([tenant_id, client_id, client_secret]):
        raise ModelServiceError("Missing Microsoft Graph credentials in environment variables.")

    try:
        import requests
    except Exception as exc:
        raise ModelServiceError("requests package is required for Microsoft Graph integration.") from exc

    response = requests.post(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        },
        timeout=15,
    )
    if response.status_code >= 400:
        raise ModelServiceError(f"Microsoft token request failed ({response.status_code}).")
    payload = response.json() if response.content else {}
    access_token = str(payload.get("access_token", "")).strip()
    if not access_token:
        raise ModelServiceError("Microsoft OAuth response did not include access_token.")
    return access_token


@api_bp.route("/integrations/status", methods=["GET"])
@jwt_required(optional=True)
def integration_status():
    status = {
        "whatsapp": {
            "configured": bool(current_app.config.get("WHATSAPP_VERIFY_TOKEN") and current_app.config.get("WHATSAPP_ACCESS_TOKEN")),
            "webhook_path": "/api/integrations/whatsapp/webhook",
        },
        "gmail": {
            "configured": bool(
                current_app.config.get("GMAIL_CLIENT_ID")
                and current_app.config.get("GMAIL_CLIENT_SECRET")
                and current_app.config.get("GMAIL_REFRESH_TOKEN")
            ),
            "pull_path": "/api/integrations/email/pull-gmail",
            "webhook_path": "/api/integrations/gmail/webhook",
        },
        "outlook": {
            "configured": bool(
                current_app.config.get("MICROSOFT_TENANT_ID")
                and current_app.config.get("MICROSOFT_CLIENT_ID")
                and current_app.config.get("MICROSOFT_CLIENT_SECRET")
            ),
            "pull_path": "/api/integrations/email/pull-outlook",
            "webhook_path": "/api/integrations/outlook/webhook",
        },
        "imap": {
            "configured": bool(
                current_app.config.get("IMAP_HOST")
                and current_app.config.get("IMAP_USERNAME")
                and current_app.config.get("IMAP_PASSWORD")
            ),
            "pull_path": "/api/integrations/email/pull-imap",
        },
        "android_sms": {
            "configured": bool(current_app.config.get("ANDROID_INGEST_TOKEN") or current_app.config.get("INTEGRATION_SHARED_TOKEN")),
            "ingest_path": "/api/integrations/message/ingest",
        },
        "browser_extension": {
            "configured": bool(current_app.config.get("BROWSER_EXTENSION_SHARED_TOKEN") or current_app.config.get("INTEGRATION_SHARED_TOKEN")),
            "scan_path": "/api/integrations/browser/scan",
            "download_path": "/api/download/browser-extension",
        },
        "file_watcher": {
            "configured": bool(current_app.config.get("FILE_WATCHER_SHARED_TOKEN") or current_app.config.get("INTEGRATION_SHARED_TOKEN")),
            "ingest_path": "/api/integrations/filesystem/ingest",
        },
    }
    return jsonify({"integrations": status})


@api_bp.route("/integrations/email/ingest", methods=["POST"])
@jwt_required(optional=True)
def ingest_email_integration():
    limit_response = _enforce_rate_limit("integration_email_ingest", upload=True)
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(current_app.config.get("INTEGRATION_SHARED_TOKEN"))
    if auth_response:
        return auth_response

    payload = request.get_json(silent=True) or {}
    subject = str(payload.get("subject", ""))
    message = str(payload.get("message", ""))
    source = str(payload.get("source", "integration_ingest"))
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

    attachments: list[dict[str, Any]] = []
    for attachment in payload.get("attachments", []) or []:
        file_bytes = _decode_base64_payload(attachment.get("file_content_base64"))
        if not file_bytes:
            continue
        attachments.append(
            {
                "filename": str(attachment.get("filename", "attachment.bin")),
                "file_bytes": file_bytes,
            }
        )

    result = _integration_scan_email(
        identity=get_jwt_identity(),
        subject=subject,
        message=message,
        attachments=attachments,
        source=source,
        metadata=metadata,
    )
    return jsonify(result), 200


@api_bp.route("/integrations/message/ingest", methods=["POST"])
@jwt_required(optional=True)
def ingest_message_integration():
    limit_response = _enforce_rate_limit("integration_message_ingest")
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(
        current_app.config.get("ANDROID_INGEST_TOKEN"),
        current_app.config.get("INTEGRATION_SHARED_TOKEN"),
    )
    if auth_response:
        return auth_response

    payload = request.get_json(silent=True) or {}
    message_text = str(payload.get("message", ""))
    source = str(payload.get("source", "android_sms_receiver"))
    metadata = {
        "sender": str(payload.get("sender", "")),
        "app": str(payload.get("app", "")),
        "received_at": str(payload.get("received_at", "")),
    }
    result = _integration_scan_message(
        identity=get_jwt_identity(),
        message_text=message_text,
        source=source,
        metadata=metadata,
    )
    return jsonify(result), 200


@api_bp.route("/integrations/browser/scan", methods=["POST"])
@jwt_required(optional=True)
def integration_browser_scan():
    limit_response = _enforce_rate_limit("integration_browser_scan")
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(
        current_app.config.get("BROWSER_EXTENSION_SHARED_TOKEN"),
        current_app.config.get("INTEGRATION_SHARED_TOKEN"),
    )
    if auth_response:
        return auth_response

    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()
    page_excerpt = str(payload.get("page_excerpt", ""))
    page_title = str(payload.get("page_title", ""))

    if not url:
        return jsonify({"error": "url is required"}), 400

    identity = get_jwt_identity()
    model_result = _get_model_service().predict_url(url)

    if page_excerpt.strip():
        try:
            text_model = _get_model_service().predict_message(page_excerpt)
            text_risk = float(text_model.get("risk_score", 0.0))
            base_risk = float(model_result.get("risk_score", 0.0))
            merged_risk = min(99.9, (0.7 * base_risk) + (0.3 * text_risk))
            model_result["risk_score"] = round(merged_risk, 2)
            details = dict(model_result.get("details", {}) or {})
            details["browser_text_analysis"] = {
                "risk_score": round(text_risk, 2),
                "verdict": text_model.get("verdict"),
            }
            model_result["details"] = details
        except Exception as exc:
            LOGGER.warning("Browser excerpt analysis failed: %s", exc)

    details = dict(model_result.get("details", {}) or {})
    details["integration"] = {
        "source": "browser_extension",
        "page_title": page_title,
    }
    model_result["details"] = details

    scan_payload = {"url": url}
    record = _persist_model_scan(identity, "url", scan_payload, model_result, f"{page_title}\n{url}".strip())
    return jsonify(_scan_response(record)), 200


@api_bp.route("/integrations/filesystem/ingest", methods=["POST"])
@jwt_required(optional=True)
def integration_filesystem_ingest():
    limit_response = _enforce_rate_limit("integration_filesystem_ingest", upload=True)
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(
        current_app.config.get("FILE_WATCHER_SHARED_TOKEN"),
        current_app.config.get("INTEGRATION_SHARED_TOKEN"),
    )
    if auth_response:
        return auth_response

    if "file" in request.files:
        uploaded = request.files["file"]
        raw = uploaded.read() if uploaded else b""
        filename = getattr(uploaded, "filename", "watched_file")
    else:
        payload = request.get_json(silent=True) or {}
        filename = str(payload.get("filename", "watched_file"))
        raw = _decode_base64_payload(payload.get("file_content_base64"))

    if not raw:
        return jsonify({"error": "file payload is required"}), 400

    identity = get_jwt_identity()
    file_sandbox = [_sandbox_analyze_attachment(raw, filename, source="filesystem_watcher")]
    model_result = _get_model_service().predict_file(raw, filename=filename)
    model_result = _apply_sandbox_to_result(model_result, "file", file_sandbox)
    details = dict(model_result.get("details", {}) or {})
    details["integration"] = {"source": "filesystem_watcher"}
    model_result["details"] = details

    scan_payload = {
        "filename": filename,
        "file_bytes": raw,
        "attachment_sandbox": file_sandbox,
    }
    record = _persist_model_scan(identity, "file", scan_payload, model_result, filename)
    return jsonify(_scan_response(record)), 200


@api_bp.route("/integrations/email/pull-imap", methods=["POST"])
@jwt_required(optional=True)
def integration_pull_imap():
    limit_response = _enforce_rate_limit("integration_pull_imap", upload=True)
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(current_app.config.get("INTEGRATION_SHARED_TOKEN"))
    if auth_response:
        return auth_response

    host = str(current_app.config.get("IMAP_HOST", "")).strip()
    username = str(current_app.config.get("IMAP_USERNAME", "")).strip()
    password = str(current_app.config.get("IMAP_PASSWORD", "")).strip()
    port = int(current_app.config.get("IMAP_PORT", 993))
    use_ssl = bool(current_app.config.get("IMAP_USE_SSL", True))
    if not all([host, username, password]):
        return jsonify({"error": "IMAP credentials are not configured"}), 400

    limit = max(1, min(25, int((request.get_json(silent=True) or {}).get("limit", 10))))
    identity = get_jwt_identity()

    import imaplib
    from email import message_from_bytes
    from email.header import decode_header, make_header

    scanned: list[dict[str, Any]] = []
    connection = None
    try:
        connection = imaplib.IMAP4_SSL(host, port) if use_ssl else imaplib.IMAP4(host, port)
        connection.login(username, password)
        connection.select("INBOX")
        status, data = connection.search(None, "UNSEEN")
        if status != "OK":
            return jsonify({"error": "unable to read inbox"}), 502

        message_ids = (data[0] or b"").split()[:limit]
        for message_id in message_ids:
            fetch_status, fetched = connection.fetch(message_id, "(RFC822)")
            if fetch_status != "OK" or not fetched:
                continue
            raw_email = fetched[0][1] if isinstance(fetched[0], tuple) and len(fetched[0]) > 1 else b""
            if not raw_email:
                continue

            msg = message_from_bytes(raw_email)
            subject = str(make_header(decode_header(msg.get("Subject", ""))))
            body_chunks: list[str] = []
            attachments: list[dict[str, Any]] = []

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = str(part.get_content_type() or "").lower()
                    disposition = str(part.get("Content-Disposition", "")).lower()
                    filename = str(part.get_filename() or "")
                    payload_bytes = part.get_payload(decode=True) or b""
                    if "attachment" in disposition and payload_bytes:
                        attachments.append({"filename": filename or "attachment.bin", "file_bytes": payload_bytes})
                    elif content_type in {"text/plain", "text/html"} and payload_bytes:
                        body_chunks.append(payload_bytes.decode("utf-8", errors="ignore"))
            else:
                payload_bytes = msg.get_payload(decode=True) or b""
                if payload_bytes:
                    body_chunks.append(payload_bytes.decode("utf-8", errors="ignore"))

            body_text = "\n".join(body_chunks).strip()
            response = _integration_scan_email(
                identity=identity,
                subject=subject,
                message=body_text,
                attachments=attachments,
                source="imap_poll",
                metadata={"provider": "imap", "mailbox": username},
            )
            scanned.append(response)
            connection.store(message_id, "+FLAGS", "\\Seen")
    finally:
        if connection is not None:
            try:
                connection.logout()
            except Exception:
                pass

    return jsonify({"scanned_count": len(scanned), "items": scanned}), 200


@api_bp.route("/integrations/email/pull-gmail", methods=["POST"])
@jwt_required(optional=True)
def integration_pull_gmail():
    limit_response = _enforce_rate_limit("integration_pull_gmail", upload=True)
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(current_app.config.get("INTEGRATION_SHARED_TOKEN"))
    if auth_response:
        return auth_response

    payload = request.get_json(silent=True) or {}
    limit = max(1, min(25, int(payload.get("limit", 10))))
    include_attachments = bool(payload.get("include_attachments", True))
    user_email = str(current_app.config.get("GMAIL_USER_EMAIL", "me")).strip() or "me"

    try:
        import requests
    except Exception as exc:
        raise ModelServiceError("requests package is required for Gmail integration.") from exc

    token = _get_google_access_token()
    identity = get_jwt_identity()
    headers = {"Authorization": f"Bearer {token}"}

    list_resp = requests.get(
        f"https://gmail.googleapis.com/gmail/v1/users/{user_email}/messages",
        headers=headers,
        params={"q": "is:unread", "maxResults": limit},
        timeout=20,
    )
    if list_resp.status_code >= 400:
        return jsonify({"error": f"gmail list request failed ({list_resp.status_code})"}), 502
    message_rows = (list_resp.json() or {}).get("messages", []) or []

    scanned: list[dict[str, Any]] = []
    for row in message_rows:
        message_id = str(row.get("id", "")).strip()
        if not message_id:
            continue
        msg_resp = requests.get(
            f"https://gmail.googleapis.com/gmail/v1/users/{user_email}/messages/{message_id}",
            headers=headers,
            params={"format": "full"},
            timeout=20,
        )
        if msg_resp.status_code >= 400:
            continue
        msg_payload = msg_resp.json() if msg_resp.content else {}
        headers_map = _extract_gmail_headers(msg_payload)
        subject = headers_map.get("subject", "")
        body_text = _extract_gmail_message_text(msg_payload)
        attachments: list[dict[str, Any]] = []

        if include_attachments:
            stack = [msg_payload.get("payload", {})]
            while stack:
                part = stack.pop()
                for nested in part.get("parts", []) or []:
                    stack.append(nested)

                filename = str(part.get("filename", "")).strip()
                attachment_id = str((part.get("body", {}) or {}).get("attachmentId", "")).strip()
                if not attachment_id:
                    continue
                attachment_resp = requests.get(
                    f"https://gmail.googleapis.com/gmail/v1/users/{user_email}/messages/{message_id}/attachments/{attachment_id}",
                    headers=headers,
                    timeout=20,
                )
                if attachment_resp.status_code >= 400:
                    continue
                attachment_payload = attachment_resp.json() if attachment_resp.content else {}
                attachment_bytes = _decode_base64_payload(attachment_payload.get("data"))
                if attachment_bytes:
                    attachments.append({"filename": filename or f"attachment_{attachment_id}.bin", "file_bytes": attachment_bytes})

        response = _integration_scan_email(
            identity=identity,
            subject=subject,
            message=body_text,
            attachments=attachments,
            source="gmail_api",
            metadata={
                "provider": "gmail",
                "message_id": message_id,
                "from": headers_map.get("from", ""),
            },
        )
        scanned.append(response)

        requests.post(
            f"https://gmail.googleapis.com/gmail/v1/users/{user_email}/messages/{message_id}/modify",
            headers={**headers, "Content-Type": "application/json"},
            json={"removeLabelIds": ["UNREAD"]},
            timeout=15,
        )

    return jsonify({"scanned_count": len(scanned), "items": scanned}), 200


@api_bp.route("/integrations/email/pull-outlook", methods=["POST"])
@jwt_required(optional=True)
def integration_pull_outlook():
    limit_response = _enforce_rate_limit("integration_pull_outlook", upload=True)
    if limit_response:
        return limit_response

    auth_response = _require_integration_token(current_app.config.get("INTEGRATION_SHARED_TOKEN"))
    if auth_response:
        return auth_response

    user_id = str(current_app.config.get("MICROSOFT_USER_ID", "")).strip()
    if not user_id:
        return jsonify({"error": "MICROSOFT_USER_ID is required"}), 400

    payload = request.get_json(silent=True) or {}
    limit = max(1, min(25, int(payload.get("limit", 10))))
    include_attachments = bool(payload.get("include_attachments", True))

    try:
        import requests
    except Exception as exc:
        raise ModelServiceError("requests package is required for Outlook integration.") from exc

    token = _get_microsoft_access_token()
    identity = get_jwt_identity()
    headers = {"Authorization": f"Bearer {token}"}

    list_resp = requests.get(
        f"https://graph.microsoft.com/v1.0/users/{user_id}/messages",
        headers=headers,
        params={
            "$top": limit,
            "$filter": "isRead eq false",
            "$select": "id,subject,bodyPreview,from,hasAttachments",
        },
        timeout=20,
    )
    if list_resp.status_code >= 400:
        return jsonify({"error": f"outlook list request failed ({list_resp.status_code})"}), 502

    scanned: list[dict[str, Any]] = []
    for message in (list_resp.json() or {}).get("value", []) or []:
        message_id = str(message.get("id", "")).strip()
        if not message_id:
            continue

        subject = str(message.get("subject", ""))
        body_text = str(message.get("bodyPreview", ""))
        sender = str((((message.get("from") or {}).get("emailAddress") or {}).get("address")) or "")
        attachments: list[dict[str, Any]] = []

        if include_attachments and bool(message.get("hasAttachments")):
            attachments_resp = requests.get(
                f"https://graph.microsoft.com/v1.0/users/{user_id}/messages/{message_id}/attachments",
                headers=headers,
                timeout=20,
            )
            if attachments_resp.status_code < 400:
                for attachment in (attachments_resp.json() or {}).get("value", []) or []:
                    if attachment.get("@odata.type") != "#microsoft.graph.fileAttachment":
                        continue
                    attachment_bytes = _decode_base64_payload(attachment.get("contentBytes"))
                    if attachment_bytes:
                        attachments.append(
                            {
                                "filename": str(attachment.get("name", "attachment.bin")),
                                "file_bytes": attachment_bytes,
                            }
                        )

        response = _integration_scan_email(
            identity=identity,
            subject=subject,
            message=body_text,
            attachments=attachments,
            source="outlook_graph",
            metadata={"provider": "outlook", "message_id": message_id, "from": sender},
        )
        scanned.append(response)

        requests.patch(
            f"https://graph.microsoft.com/v1.0/users/{user_id}/messages/{message_id}",
            headers={**headers, "Content-Type": "application/json"},
            json={"isRead": True},
            timeout=15,
        )

    return jsonify({"scanned_count": len(scanned), "items": scanned}), 200


@api_bp.route("/integrations/gmail/webhook", methods=["POST"])
def integration_gmail_webhook():
    limit_response = _enforce_rate_limit("integration_gmail_webhook")
    if limit_response:
        return limit_response

    expected_token = str(current_app.config.get("GMAIL_PUSH_WEBHOOK_TOKEN", "")).strip()
    if expected_token:
        auth_response = _require_integration_token(expected_token)
        if auth_response:
            return auth_response

    payload = request.get_json(silent=True) or {}
    message = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
    encoded_data = str(message.get("data", ""))
    decoded: dict[str, Any] = {}
    if encoded_data:
        decoded_bytes = _decode_base64_payload(encoded_data)
        if decoded_bytes:
            try:
                decoded = json.loads(decoded_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                decoded = {"raw": decoded_bytes.decode("utf-8", errors="ignore")}

    return jsonify(
        {
            "status": "acknowledged",
            "subscription": message.get("subscription"),
            "payload": decoded,
        }
    ), 200


@api_bp.route("/integrations/outlook/webhook", methods=["GET", "POST"])
def integration_outlook_webhook():
    validation_token = request.args.get("validationToken")
    if validation_token:
        return Response(validation_token, mimetype="text/plain")

    limit_response = _enforce_rate_limit("integration_outlook_webhook")
    if limit_response:
        return limit_response

    payload = request.get_json(silent=True) or {}
    notifications = payload.get("value", []) if isinstance(payload.get("value"), list) else []
    expected = str(current_app.config.get("MICROSOFT_GRAPH_WEBHOOK_SECRET", "")).strip()
    if expected and notifications:
        if any(not hmac.compare_digest(str(item.get("clientState", "")), expected) for item in notifications):
            return jsonify({"error": "invalid outlook webhook secret"}), 401

    return jsonify({"status": "acknowledged", "notifications": len(notifications)}), 200


@api_bp.route("/integrations/whatsapp/webhook", methods=["GET", "POST"])
def integration_whatsapp_webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode", "")
        token = request.args.get("hub.verify_token", "")
        challenge = request.args.get("hub.challenge", "")
        expected = str(current_app.config.get("WHATSAPP_VERIFY_TOKEN", "")).strip()

        if mode == "subscribe" and expected and hmac.compare_digest(token, expected):
            return Response(challenge, mimetype="text/plain")
        return jsonify({"error": "verification failed"}), 403

    limit_response = _enforce_rate_limit("integration_whatsapp_webhook")
    if limit_response:
        return limit_response

    secret = str(current_app.config.get("WHATSAPP_APP_SECRET", "")).strip()
    signature = str(request.headers.get("X-Hub-Signature-256", ""))
    raw_payload = request.get_data(cache=True) or b""
    if not _verify_hub_signature(raw_payload, secret, signature):
        return jsonify({"error": "invalid whatsapp signature"}), 401

    payload = request.get_json(silent=True) or {}
    messages = _collect_whatsapp_messages(payload)
    identity = None
    scanned_items = []
    fetch_media = bool(current_app.config.get("WHATSAPP_FETCH_MEDIA", False))

    for item in messages:
        attachments: list[dict[str, Any]] = []
        attachment_meta = item.get("attachment") if isinstance(item.get("attachment"), dict) else None
        if fetch_media and attachment_meta:
            media_bytes, media_name, status = _fetch_whatsapp_media(str(attachment_meta.get("media_id", "")))
            if media_bytes:
                attachments.append({"filename": media_name or attachment_meta.get("filename", "media.bin"), "file_bytes": media_bytes})
            attachment_meta["fetch_status"] = status

        result = _integration_scan_message(
            identity=identity,
            message_text=str(item.get("text", "")),
            source="whatsapp_business_webhook",
            metadata={
                "message_id": item.get("message_id"),
                "from": item.get("from"),
                "timestamp": item.get("timestamp"),
                "type": item.get("type"),
                "contact_name": item.get("contact_name"),
                "attachment": attachment_meta,
            },
            attachments=attachments,
        )
        scanned_items.append(result)

    return jsonify({"received_messages": len(messages), "scanned": len(scanned_items), "items": scanned_items}), 200


@api_bp.route("/scan/email", methods=["POST"])
@jwt_required(optional=True)
def scan_email():
    limit_response = _enforce_rate_limit("scan_email", upload=True)
    if limit_response:
        return limit_response

    if str(request.mimetype or "").startswith("multipart/form-data"):
        subject = str(request.form.get("subject", ""))
        message = str(request.form.get("message", ""))
        attachments = []
        for uploaded in request.files.getlist("attachments"):
            raw = uploaded.read() if uploaded else b""
            if not raw:
                continue
            attachments.append(
                {
                    "filename": getattr(uploaded, "filename", "attachment"),
                    "file_bytes": raw,
                }
            )
    else:
        payload = request.get_json(silent=True) or {}
        subject = str(payload.get("subject", ""))
        message = str(payload.get("message", ""))
        attachments = []

    attachment_sandbox = [
        _sandbox_analyze_attachment(
            attachment.get("file_bytes", b""),
            attachment.get("filename", "attachment"),
            source="email_attachment",
        )
        for attachment in attachments
    ]

    if not subject.strip() and not message.strip() and attachments:
        subject = "[Attachment-only email]"
        message = "No email body provided. Risk score includes attachment sandbox signals."

    identity = get_jwt_identity()

    model_result = _get_model_service().predict_email(subject, message)
    model_result = _apply_sandbox_to_result(model_result, "email", attachment_sandbox)
    scan_payload = {
        "subject": subject,
        "message": message,
        "attachments": attachments,
        "attachment_sandbox": attachment_sandbox,
    }
    model_result, policy = _build_scan_result(
        "email",
        scan_payload,
        model_result,
    )
    sanitized_payload = _sanitize_sample_payload("email", scan_payload)
    model_result = _maybe_escalate(identity, "email", sanitized_payload, model_result, policy)

    record = _persist_scan(identity, "email", f"{subject}\n{message}", model_result, policy)
    return jsonify(_scan_response(record)), 200


@api_bp.route("/scan/message", methods=["POST"])
@jwt_required(optional=True)
def scan_message():
    limit_response = _enforce_rate_limit("scan_message")
    if limit_response:
        return limit_response

    payload = request.get_json(silent=True) or {}
    message_text = str(payload.get("message", ""))
    identity = get_jwt_identity()

    model_result = _get_model_service().predict_message(message_text)
    model_result, policy = _build_scan_result("message", {"message": message_text}, model_result)
    model_result = _maybe_escalate(identity, "message", {"message": message_text}, model_result, policy)

    record = _persist_scan(identity, "message", message_text, model_result, policy)
    return jsonify(_scan_response(record)), 200


@api_bp.route("/scan/url", methods=["POST"])
@jwt_required(optional=True)
def scan_url():
    limit_response = _enforce_rate_limit("scan_url")
    if limit_response:
        return limit_response

    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", ""))
    identity = get_jwt_identity()

    model_result = _get_model_service().predict_url(url)
    model_result, policy = _build_scan_result("url", {"url": url}, model_result)
    model_result = _maybe_escalate(identity, "url", {"url": url}, model_result, policy)

    record = _persist_scan(identity, "url", url, model_result, policy)
    return jsonify(_scan_response(record)), 200


@api_bp.route("/scan/file", methods=["POST"])
@jwt_required(optional=True)
def scan_file():
    limit_response = _enforce_rate_limit("scan_file", upload=True)
    if limit_response:
        return limit_response

    if "file" not in request.files:
        return jsonify({"error": "multipart form file is required under key 'file'"}), 400

    uploaded = request.files["file"]
    raw = uploaded.read() if uploaded else b""
    filename = getattr(uploaded, "filename", "uploaded_file")
    identity = get_jwt_identity()
    file_sandbox = [_sandbox_analyze_attachment(raw, filename, source="file_upload")]

    model_result = _get_model_service().predict_file(raw, filename=filename)
    model_result = _apply_sandbox_to_result(model_result, "file", file_sandbox)
    scan_payload = {
        "filename": filename,
        "file_bytes": raw,
        "attachment_sandbox": file_sandbox,
    }
    model_result, policy = _build_scan_result("file", scan_payload, model_result)
    sanitized_payload = _sanitize_sample_payload("file", scan_payload)
    model_result = _maybe_escalate(identity, "file", sanitized_payload, model_result, policy)

    record = _persist_scan(identity, "file", filename, model_result, policy)
    return jsonify(_scan_response(record)), 200


@api_bp.route("/scan/fraud", methods=["POST"])
@jwt_required(optional=True)
def scan_fraud():
    limit_response = _enforce_rate_limit("scan_fraud")
    if limit_response:
        return limit_response

    payload = request.get_json(silent=True) or {}
    identity = get_jwt_identity()

    model_result = _get_model_service().predict_fraud(payload)
    model_result, policy = _build_scan_result("fraud", {"transaction": payload}, model_result)
    model_result = _maybe_escalate(identity, "fraud", {"transaction": payload}, model_result, policy)

    excerpt = json.dumps(
        {
            "type": payload.get("type"),
            "amount": payload.get("amount"),
            "nameOrig": payload.get("nameOrig"),
            "nameDest": payload.get("nameDest"),
        }
    )
    record = _persist_scan(identity, "fraud", excerpt, model_result, policy)
    return jsonify(_scan_response(record)), 200


@api_bp.route("/dashboard/summary", methods=["GET"])
@jwt_required()
def dashboard_summary():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    summary = _get_store().dashboard_summary(
        user["id"],
        low_confidence_threshold=_get_thresholds().uncertain_confidence,
    )
    return jsonify(summary)


@api_bp.route("/dashboard/history", methods=["GET"])
@jwt_required()
def dashboard_history():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    limit = max(1, min(200, int(request.args.get("limit", 50))))
    items = _get_store().get_scan_history(user["id"], limit=limit)
    return jsonify({"items": [_scan_response(item) for item in items]})


@api_bp.route("/alerts", methods=["GET"])
@jwt_required()
def list_alerts():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    alerts = _get_store().get_alerts(user["id"], limit=100)
    return jsonify({"items": [_serialize_alert(alert) for alert in alerts]})


@api_bp.route("/alerts/<alert_id>/ack", methods=["PATCH"])
@jwt_required()
def acknowledge_alert(alert_id: str):
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    ok = _get_store().acknowledge_alert(alert_id, user["id"])
    if not ok:
        return jsonify({"error": "alert not found"}), 404
    return jsonify({"message": "alert acknowledged", "alert_id": alert_id})


@api_bp.route("/stream/alerts", methods=["GET"])
def stream_alerts():
    token = request.args.get("token", "")
    user_id = None

    if token:
        try:
            decoded = decode_token(token)
            user_id = decoded.get("sub")
        except Exception:
            return jsonify({"error": "invalid stream token"}), 401

    since_id = request.args.get("since", "")

    def event_generator(last_seen_id: str):
        cursor = last_seen_id or None
        while True:
            batch = _get_store().get_new_alerts(since_id=cursor, user_id=user_id, limit=20)
            if batch:
                for alert in batch:
                    payload = _serialize_alert(alert)
                    yield f"event: alert\ndata: {json.dumps(payload)}\n\n"
                cursor = batch[-1]["id"]
            else:
                yield "event: ping\ndata: {}\n\n"
            time.sleep(3)

    return Response(event_generator(since_id), mimetype="text/event-stream")


@api_bp.route("/models/status", methods=["GET"])
def model_status():
    return jsonify({"models": _get_model_service().get_status()})


@api_bp.route("/review/queue", methods=["GET"])
@jwt_required()
def get_review_queue():
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    status = request.args.get("status", "pending")
    scan_type = request.args.get("scan_type")
    limit = max(1, min(500, int(request.args.get("limit", 100))))

    items = _get_store().list_active_learning_queue(
        status=status,
        scan_type=scan_type,
        limit=limit,
        user_id=user["id"],
    )
    return jsonify({"items": items})


@api_bp.route("/review/queue/<item_id>", methods=["PATCH"])
@jwt_required()
def resolve_review_item(item_id: str):
    user = _current_user()
    if user is None:
        return jsonify({"error": "user not found"}), 404

    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    status = str(payload.get("status", "reviewed")).strip() or "reviewed"

    if not label:
        return jsonify({"error": "label is required"}), 400

    updated = _get_store().resolve_active_learning_item(
        item_id,
        label=label,
        reviewer_notes=notes,
        status=status,
        user_id=user["id"],
    )
    if updated is None:
        return jsonify({"error": "queue item not found"}), 404

    return jsonify({"item": updated})


@api_bp.route("/download/browser-extension", methods=["GET"])
def download_browser_extension():
    extension_root = Path(current_app.config.get("BROWSER_EXTENSION_DIR", "")).resolve()
    if not extension_root.exists() or not extension_root.is_dir():
        return jsonify({"error": "browser extension directory not found"}), 404

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in extension_root.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(extension_root).as_posix()
            zf.write(path, arcname=f"browser-extension/{rel}")

    archive.seek(0)
    return send_file(
        archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name="scam_defender_browser_extension.zip",
    )


@api_bp.route("/download/android", methods=["GET"])
def download_android_bundle():
    release_dir = Path(current_app.config.get("ANDROID_RELEASE_DIR", "")).resolve()
    if not release_dir.exists() or not release_dir.is_dir():
        return jsonify({"error": "android release directory not found"}), 404

    apk_files = sorted(
        [path for path in release_dir.glob("*.apk") if path.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not apk_files:
        return jsonify({"error": "no Android APK artifact found in release directory"}), 404

    latest_apk = apk_files[0]
    return send_file(
        latest_apk,
        mimetype="application/vnd.android.package-archive",
        as_attachment=True,
        download_name=latest_apk.name,
    )


@api_bp.route("/download/android/<filename>", methods=["GET"])
def download_android_named(filename: str):
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".apk"):
        return jsonify({"error": "only APK downloads are allowed"}), 400

    release_dir = Path(current_app.config.get("ANDROID_RELEASE_DIR", "")).resolve()
    apk_path = (release_dir / safe_name).resolve()
    if not str(apk_path).startswith(str(release_dir)) or not apk_path.exists():
        return jsonify({"error": "apk not found"}), 404

    return send_file(
        apk_path,
        mimetype="application/vnd.android.package-archive",
        as_attachment=True,
        download_name=apk_path.name,
    )


@api_bp.route("/download/source", methods=["GET"])
def download_source_bundle():
    root = Path(current_app.config["PROJECT_ROOT"]).resolve()
    archive = io.BytesIO()

    excluded_part_dirs = {
        ".git",
        "node_modules",
        ".projVenv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".idea",
        ".vscode",
    }
    excluded_prefixes = {"frontend/dist/", "app/data/"}
    excluded_suffixes = {".db", ".log", ".pyc", ".pyo"}
    excluded_files = {".env"}

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_dir():
                continue

            rel_path = path.relative_to(root).as_posix()
            if any(part in excluded_part_dirs for part in Path(rel_path).parts):
                continue
            if any(rel_path.startswith(prefix) for prefix in excluded_prefixes):
                continue
            if path.name in excluded_files:
                continue
            if path.suffix.lower() in excluded_suffixes:
                continue

            zf.write(path, arcname=f"scam_defender/{rel_path}")

    archive.seek(0)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    download_name = f"scam_defender_source_{timestamp}.zip"

    return send_file(
        archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


if __name__ == "__main__":
    from app import create_app

    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
