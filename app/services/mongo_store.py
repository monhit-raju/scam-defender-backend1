from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


def parse_object_id(value: str | ObjectId | None) -> ObjectId | None:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def to_json_safe(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    return value


def serialize_document(document: dict | None) -> dict | None:
    if document is None:
        return None
    payload = dict(document)
    payload["id"] = str(payload.pop("_id"))
    return to_json_safe(payload)


@dataclass(slots=True)
class ScanPersistenceInput:
    user_id: ObjectId | None
    scan_type: str
    input_excerpt: str
    verdict: str
    severity: str
    confidence: float
    risk_score: float
    details: dict[str, Any]
    gating: dict[str, Any]


class MongoStore:
    def __init__(self, mongo_uri: str, database_name: str):
        self.client = MongoClient(
            mongo_uri,
            uuidRepresentation="standard",
            serverSelectionTimeoutMS=2000,
        )
        self.db = self.client[database_name]

        self.users: Collection = self.db["users"]
        self.user_profiles: Collection = self.db["user_profiles"]
        self.privacy_settings: Collection = self.db["privacy_settings"]
        self.scan_records: Collection = self.db["scan_records"]
        self.threat_alerts: Collection = self.db["threat_alerts"]
        self.feedback: Collection = self.db["feedback"]
        self.active_learning_queue: Collection = self.db["active_learning_queue"]
        self.cloud_verification_queue: Collection = self.db["cloud_verification_queue"]

        # Clean up any problematic data before creating indexes
        cleanup_stats = self.cleanup_duplicate_emails()
        if cleanup_stats["removed_duplicates"] > 0 or cleanup_stats["null_emails_cleaned"] > 0:
            print(f"Database cleanup completed: {cleanup_stats}")

        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        try:
            # Handle existing email index issues
            existing_indexes = list(self.users.list_indexes())
            email_index_exists = any(idx.get('name', '').startswith('email') for idx in existing_indexes)

            if email_index_exists:
                # Drop existing email index if it exists to avoid conflicts
                try:
                    self.users.drop_index("email_1")
                except Exception:
                    pass  # Index might not exist with this name

            # Create new email index with proper filtering
            self.users.create_index(
                [("email", ASCENDING)],
                unique=True,
                partialFilterExpression={"email": {"$exists": True, "$nin": [None, ""]}},
                name="email_unique_filtered"
            )
        except Exception as e:
            # Log the error but don't crash - indexes are not critical for basic functionality
            print(f"Warning: Could not create email index: {e}")

        # Create other indexes with error handling
        try:
            self.user_profiles.create_index([("user_id", ASCENDING)], unique=True)
        except Exception as e:
            print(f"Warning: Could not create user_profiles index: {e}")

        try:
            self.privacy_settings.create_index([("user_id", ASCENDING)], unique=True)
        except Exception as e:
            print(f"Warning: Could not create privacy_settings index: {e}")

        try:
            self.scan_records.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
            self.scan_records.create_index([("scan_type", ASCENDING), ("created_at", DESCENDING)])
            self.scan_records.create_index([("gating.action", ASCENDING), ("created_at", DESCENDING)])
        except Exception as e:
            print(f"Warning: Could not create scan_records indexes: {e}")

        try:
            self.threat_alerts.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
            self.threat_alerts.create_index([("scan_record_id", ASCENDING)], unique=True)
            self.threat_alerts.create_index([("acknowledged", ASCENDING), ("created_at", DESCENDING)])
        except Exception as e:
            print(f"Warning: Could not create threat_alerts indexes: {e}")

        try:
            self.feedback.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        except Exception as e:
            print(f"Warning: Could not create feedback index: {e}")

        try:
            self.active_learning_queue.create_index([("status", ASCENDING), ("priority", DESCENDING), ("created_at", ASCENDING)])
        except Exception as e:
            print(f"Warning: Could not create active_learning_queue index: {e}")

        try:
            self.cloud_verification_queue.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        except Exception as e:
            print(f"Warning: Could not create cloud_verification_queue index: {e}")

    def cleanup_duplicate_emails(self) -> dict:
        """Clean up duplicate email entries, keeping the most recent one."""
        cleanup_stats = {"removed_duplicates": 0, "null_emails_cleaned": 0}

        try:
            # Remove documents with null emails that might cause index issues
            result = self.users.delete_many({"email": None})
            cleanup_stats["null_emails_cleaned"] = result.deleted_count

            # Find and handle duplicate emails (keeping the most recent)
            pipeline = [
                {"$match": {"email": {"$exists": True, "$ne": None, "$ne": ""}}},
                {"$group": {
                    "_id": "$email",
                    "docs": {"$push": {"_id": "$_id", "created_at": "$created_at"}},
                    "count": {"$sum": 1}
                }},
                {"$match": {"count": {"$gt": 1}}}
            ]

            duplicates = list(self.users.aggregate(pipeline))

            for dup in duplicates:
                # Sort by creation date, keep the most recent
                docs = sorted(dup["docs"], key=lambda x: x["created_at"], reverse=True)
                # Remove all but the most recent
                to_remove = [doc["_id"] for doc in docs[1:]]
                if to_remove:
                    self.users.delete_many({"_id": {"$in": to_remove}})
                    cleanup_stats["removed_duplicates"] += len(to_remove)

        except Exception as e:
            print(f"Warning: Error during email cleanup: {e}")

        return cleanup_stats

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    @staticmethod
    def _default_profile(user_id: ObjectId) -> dict[str, Any]:
        now = utcnow()
        return {
            "user_id": user_id,
            "avatar_url": "",
            "role": "Security Analyst",
            "company": "Independent",
            "location": "",
            "phone": "",
            "bio": "",
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def _default_privacy(user_id: ObjectId) -> dict[str, Any]:
        now = utcnow()
        return {
            "user_id": user_id,
            "two_factor_enabled": False,
            "email_alerts": True,
            "sms_alerts": False,
            "share_anonymized_analytics": True,
            "data_retention_days": 90,
            "profile_visibility": "private",
            "created_at": now,
            "updated_at": now,
        }

    def create_user(self, email: str, password_hash: str, full_name: str) -> dict:
        now = utcnow()
        user_doc = {
            "email": email.strip().lower(),
            "password_hash": password_hash,
            "full_name": full_name.strip() or "Scam Defender User",
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = self.users.insert_one(user_doc)
        except DuplicateKeyError as exc:
            raise ValueError("user already exists") from exc

        user_id = result.inserted_id
        self.user_profiles.update_one({"user_id": user_id}, {"$setOnInsert": self._default_profile(user_id)}, upsert=True)
        self.privacy_settings.update_one({"user_id": user_id}, {"$setOnInsert": self._default_privacy(user_id)}, upsert=True)
        return serialize_document(self.users.find_one({"_id": user_id}))

    def get_user_by_email(self, email: str) -> dict | None:
        return serialize_document(self.users.find_one({"email": email.strip().lower()}))

    def get_user_by_id(self, user_id: str | ObjectId | None) -> dict | None:
        oid = parse_object_id(user_id)
        if oid is None:
            return None
        return serialize_document(self.users.find_one({"_id": oid}))

    def _get_raw_user(self, user_id: str | ObjectId | None) -> dict | None:
        oid = parse_object_id(user_id)
        if oid is None:
            return None
        return self.users.find_one({"_id": oid})

    def get_profile(self, user_id: str | ObjectId) -> dict:
        oid = parse_object_id(user_id)
        if oid is None:
            raise ValueError("invalid user id")

        profile = self.user_profiles.find_one({"user_id": oid})
        if profile is None:
            self.user_profiles.insert_one(self._default_profile(oid))
            profile = self.user_profiles.find_one({"user_id": oid})
        return serialize_document(profile)

    def get_privacy(self, user_id: str | ObjectId) -> dict:
        oid = parse_object_id(user_id)
        if oid is None:
            raise ValueError("invalid user id")

        settings = self.privacy_settings.find_one({"user_id": oid})
        if settings is None:
            self.privacy_settings.insert_one(self._default_privacy(oid))
            settings = self.privacy_settings.find_one({"user_id": oid})
        return serialize_document(settings)

    def update_profile(self, user_id: str | ObjectId, payload: dict[str, Any]) -> dict:
        oid = parse_object_id(user_id)
        if oid is None:
            raise ValueError("invalid user id")

        now = utcnow()
        existing_profile = self.user_profiles.find_one({"user_id": oid}) or self._default_profile(oid)
        user_updates: dict[str, Any] = {}
        if "full_name" in payload:
            full_name = str(payload.get("full_name", "")).strip()
            if full_name:
                user_updates["full_name"] = full_name

        if user_updates:
            user_updates["updated_at"] = now
            self.users.update_one({"_id": oid}, {"$set": user_updates})

        profile_updates = {
            "avatar_url": str(payload.get("avatar_url", existing_profile.get("avatar_url", ""))),
            "role": str(payload.get("role", existing_profile.get("role", "Security Analyst"))),
            "company": str(payload.get("company", existing_profile.get("company", "Independent"))),
            "location": str(payload.get("location", existing_profile.get("location", ""))),
            "phone": str(payload.get("phone", existing_profile.get("phone", ""))),
            "bio": str(payload.get("bio", existing_profile.get("bio", ""))),
            "updated_at": now,
        }

        self.user_profiles.update_one(
            {"user_id": oid},
            {"$set": profile_updates, "$setOnInsert": {"created_at": now, "user_id": oid}},
            upsert=True,
        )

        user_doc = self._get_raw_user(oid)
        profile_doc = self.user_profiles.find_one({"user_id": oid})
        privacy_doc = self.privacy_settings.find_one({"user_id": oid}) or self._default_privacy(oid)
        return self._compose_user_payload(user_doc, profile_doc, privacy_doc)

    def update_privacy(self, user_id: str | ObjectId, payload: dict[str, Any]) -> dict:
        oid = parse_object_id(user_id)
        if oid is None:
            raise ValueError("invalid user id")

        now = utcnow()
        existing = self.privacy_settings.find_one({"user_id": oid}) or self._default_privacy(oid)
        retention_days = max(7, min(3650, int(payload.get("data_retention_days", existing.get("data_retention_days", 90)))))

        updates = {
            "two_factor_enabled": bool(payload.get("two_factor_enabled", existing.get("two_factor_enabled", False))),
            "email_alerts": bool(payload.get("email_alerts", existing.get("email_alerts", True))),
            "sms_alerts": bool(payload.get("sms_alerts", existing.get("sms_alerts", False))),
            "share_anonymized_analytics": bool(
                payload.get("share_anonymized_analytics", existing.get("share_anonymized_analytics", True))
            ),
            "data_retention_days": retention_days,
            "profile_visibility": str(payload.get("profile_visibility", existing.get("profile_visibility", "private"))),
            "updated_at": now,
        }
        self.privacy_settings.update_one(
            {"user_id": oid},
            {"$set": updates, "$setOnInsert": {"created_at": now, "user_id": oid}},
            upsert=True,
        )
        return serialize_document(self.privacy_settings.find_one({"user_id": oid}))

    def compose_user(self, user_id: str | ObjectId) -> dict | None:
        oid = parse_object_id(user_id)
        if oid is None:
            return None

        user_doc = self._get_raw_user(oid)
        if user_doc is None:
            return None
        profile_doc = self.user_profiles.find_one({"user_id": oid}) or self._default_profile(oid)
        privacy_doc = self.privacy_settings.find_one({"user_id": oid}) or self._default_privacy(oid)
        return self._compose_user_payload(user_doc, profile_doc, privacy_doc)

    def _compose_user_payload(self, user_doc: dict, profile_doc: dict, privacy_doc: dict) -> dict:
        return {
            "id": str(user_doc["_id"]),
            "email": user_doc["email"],
            "full_name": user_doc.get("full_name", "Scam Defender User"),
            "is_active": bool(user_doc.get("is_active", True)),
            "created_at": to_json_safe(user_doc.get("created_at")),
            "updated_at": to_json_safe(user_doc.get("updated_at")),
            "profile": {
                "avatar_url": str(profile_doc.get("avatar_url", "")),
                "role": str(profile_doc.get("role", "Security Analyst")),
                "company": str(profile_doc.get("company", "Independent")),
                "location": str(profile_doc.get("location", "")),
                "phone": str(profile_doc.get("phone", "")),
                "bio": str(profile_doc.get("bio", "")),
            },
            "privacy": {
                "two_factor_enabled": bool(privacy_doc.get("two_factor_enabled", False)),
                "email_alerts": bool(privacy_doc.get("email_alerts", True)),
                "sms_alerts": bool(privacy_doc.get("sms_alerts", False)),
                "share_anonymized_analytics": bool(privacy_doc.get("share_anonymized_analytics", True)),
                "data_retention_days": int(privacy_doc.get("data_retention_days", 90)),
                "profile_visibility": str(privacy_doc.get("profile_visibility", "private")),
            },
        }

    def create_scan_record(self, payload: ScanPersistenceInput) -> dict:
        now = utcnow()
        scan_doc = {
            "user_id": payload.user_id,
            "scan_type": payload.scan_type,
            "input_excerpt": payload.input_excerpt,
            "verdict": payload.verdict,
            "severity": payload.severity,
            "confidence": float(payload.confidence),
            "risk_score": float(payload.risk_score),
            "details": payload.details,
            "gating": payload.gating,
            "created_at": now,
            "updated_at": now,
        }
        inserted = self.scan_records.insert_one(scan_doc)
        scan_id = inserted.inserted_id
        scan = self.scan_records.find_one({"_id": scan_id})
        return serialize_document(scan)

    def create_alert(self, scan_record_id: str | ObjectId, user_id: str | ObjectId | None, severity: str, title: str, message: str) -> dict:
        scan_oid = parse_object_id(scan_record_id)
        user_oid = parse_object_id(user_id)
        now = utcnow()
        alert_doc = {
            "scan_record_id": scan_oid,
            "user_id": user_oid,
            "severity": severity,
            "title": title,
            "message": message,
            "acknowledged": False,
            "created_at": now,
            "updated_at": now,
        }
        try:
            inserted = self.threat_alerts.insert_one(alert_doc)
            return serialize_document(self.threat_alerts.find_one({"_id": inserted.inserted_id}))
        except DuplicateKeyError:
            existing = self.threat_alerts.find_one({"scan_record_id": scan_oid})
            return serialize_document(existing)

    def get_alerts(self, user_id: str | ObjectId | None, limit: int = 100) -> list[dict]:
        query: dict[str, Any] = {}
        oid = parse_object_id(user_id)
        if oid is not None:
            query["user_id"] = oid
        rows = list(self.threat_alerts.find(query).sort("created_at", DESCENDING).limit(limit))
        return [serialize_document(row) for row in rows]

    def acknowledge_alert(self, alert_id: str | ObjectId, user_id: str | ObjectId | None) -> bool:
        alert_oid = parse_object_id(alert_id)
        if alert_oid is None:
            return False

        query: dict[str, Any] = {"_id": alert_oid}
        user_oid = parse_object_id(user_id)
        if user_id is not None:
            query["user_id"] = user_oid

        result = self.threat_alerts.update_one(
            query,
            {"$set": {"acknowledged": True, "updated_at": utcnow()}},
        )
        return result.matched_count > 0

    def get_scan_history(self, user_id: str | ObjectId | None, limit: int = 50) -> list[dict]:
        query: dict[str, Any] = {}
        oid = parse_object_id(user_id)
        if oid is not None:
            query["user_id"] = oid
        rows = list(self.scan_records.find(query).sort("created_at", DESCENDING).limit(limit))
        return [serialize_document(row) for row in rows]

    def dashboard_summary(self, user_id: str | ObjectId, low_confidence_threshold: float = 0.7) -> dict:
        oid = parse_object_id(user_id)
        if oid is None:
            return {
                "totals": {"total_scans": 0, "threat_scans": 0, "recent_scans_7d": 0, "open_alerts": 0},
                "by_type": {},
                "by_severity": {},
                "ops_metrics": {
                    "low_confidence_calls": 0,
                    "uncertain_route_calls": 0,
                    "anomaly_flagged_calls": 0,
                    "cloud_escalations": 0,
                },
            }

        now = utcnow()
        seven_days_ago = now - dt.timedelta(days=7)

        base_query = {"user_id": oid}
        total_scans = self.scan_records.count_documents(base_query)
        threat_scans = self.scan_records.count_documents({**base_query, "severity": {"$in": ["high", "critical"]}})
        recent_scans = self.scan_records.count_documents({**base_query, "created_at": {"$gte": seven_days_ago}})
        open_alerts = self.threat_alerts.count_documents({"user_id": oid, "acknowledged": False})

        by_type = {}
        by_severity = {}
        for row in self.scan_records.aggregate(
            [
                {"$match": base_query},
                {"$group": {"_id": "$scan_type", "count": {"$sum": 1}}},
            ]
        ):
            by_type[str(row["_id"])] = int(row["count"])

        for row in self.scan_records.aggregate(
            [
                {"$match": base_query},
                {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            ]
        ):
            by_severity[str(row["_id"])] = int(row["count"])

        low_conf_calls = self.scan_records.count_documents(
            {**base_query, "confidence": {"$lt": float(low_confidence_threshold)}}
        )
        uncertain_route_calls = self.scan_records.count_documents({**base_query, "gating.stage": "uncertain"})
        anomaly_calls = self.scan_records.count_documents({**base_query, "details.anomaly.flagged": True})
        cloud_escalations = self.cloud_verification_queue.count_documents({**base_query, "created_at": {"$gte": seven_days_ago}})

        return {
            "totals": {
                "total_scans": int(total_scans),
                "threat_scans": int(threat_scans),
                "recent_scans_7d": int(recent_scans),
                "open_alerts": int(open_alerts),
            },
            "by_type": by_type,
            "by_severity": by_severity,
            "ops_metrics": {
                "low_confidence_calls": int(low_conf_calls),
                "uncertain_route_calls": int(uncertain_route_calls),
                "anomaly_flagged_calls": int(anomaly_calls),
                "cloud_escalations": int(cloud_escalations),
            },
        }

    def insert_feedback(self, user_id: str | ObjectId | None, payload: dict[str, Any]) -> dict:
        oid = parse_object_id(user_id)
        now = utcnow()
        document = {
            "user_id": oid,
            "category": str(payload.get("category", "general")),
            "rating": max(1, min(5, int(payload.get("rating", 5)))),
            "subject": str(payload.get("subject", "")),
            "message": str(payload.get("message", "")).strip(),
            "contact_email": str(payload.get("contact_email", "")),
            "created_at": now,
            "updated_at": now,
        }
        inserted = self.feedback.insert_one(document)
        return serialize_document(self.feedback.find_one({"_id": inserted.inserted_id}))

    def list_feedback(self, user_id: str | ObjectId | None, limit: int = 100) -> list[dict]:
        query: dict[str, Any] = {}
        oid = parse_object_id(user_id)
        if oid is not None:
            query["user_id"] = oid
        rows = list(self.feedback.find(query).sort("created_at", DESCENDING).limit(limit))
        return [serialize_document(row) for row in rows]

    def enqueue_active_learning(
        self,
        user_id: str | ObjectId | None,
        scan_type: str,
        sample_payload: dict[str, Any],
        model_output: dict[str, Any],
        reason: str,
        priority: int = 50,
        status: str = "pending",
    ) -> dict:
        oid = parse_object_id(user_id)
        now = utcnow()
        document = {
            "user_id": oid,
            "scan_type": scan_type,
            "sample_payload": sample_payload,
            "model_output": model_output,
            "reason": reason,
            "priority": int(priority),
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        inserted = self.active_learning_queue.insert_one(document)
        return serialize_document(self.active_learning_queue.find_one({"_id": inserted.inserted_id}))

    def list_active_learning_queue(
        self,
        status: str | None = "pending",
        scan_type: str | None = None,
        limit: int = 100,
        user_id: str | ObjectId | None = None,
    ) -> list[dict]:
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if scan_type:
            query["scan_type"] = scan_type
        user_oid = parse_object_id(user_id)
        if user_id is not None:
            query["user_id"] = user_oid

        rows = list(
            self.active_learning_queue.find(query)
            .sort([("priority", DESCENDING), ("created_at", ASCENDING)])
            .limit(limit)
        )
        return [serialize_document(row) for row in rows]

    def resolve_active_learning_item(
        self,
        item_id: str | ObjectId,
        label: str,
        reviewer_notes: str = "",
        status: str = "reviewed",
        user_id: str | ObjectId | None = None,
    ) -> dict | None:
        oid = parse_object_id(item_id)
        if oid is None:
            return None

        query: dict[str, Any] = {"_id": oid}
        user_oid = parse_object_id(user_id)
        if user_id is not None:
            query["user_id"] = user_oid

        self.active_learning_queue.update_one(
            query,
            {
                "$set": {
                    "status": status,
                    "resolved_label": str(label),
                    "reviewer_notes": str(reviewer_notes),
                    "updated_at": utcnow(),
                }
            },
        )
        return serialize_document(self.active_learning_queue.find_one(query))

    def enqueue_cloud_verification(
        self,
        user_id: str | ObjectId | None,
        scan_type: str,
        sample_payload: dict[str, Any],
        reason: str,
        status: str = "queued",
    ) -> dict:
        oid = parse_object_id(user_id)
        now = utcnow()
        document = {
            "user_id": oid,
            "scan_type": scan_type,
            "sample_payload": sample_payload,
            "reason": reason,
            "status": status,
            "cloud_response": None,
            "created_at": now,
            "updated_at": now,
        }
        inserted = self.cloud_verification_queue.insert_one(document)
        return serialize_document(self.cloud_verification_queue.find_one({"_id": inserted.inserted_id}))

    def update_cloud_verification(self, item_id: str | ObjectId, status: str, cloud_response: dict[str, Any] | None = None) -> dict | None:
        oid = parse_object_id(item_id)
        if oid is None:
            return None

        self.cloud_verification_queue.update_one(
            {"_id": oid},
            {
                "$set": {
                    "status": status,
                    "cloud_response": cloud_response,
                    "updated_at": utcnow(),
                }
            },
        )
        return serialize_document(self.cloud_verification_queue.find_one({"_id": oid}))

    def get_new_alerts(self, since_id: str | ObjectId | None = None, user_id: str | ObjectId | None = None, limit: int = 20) -> list[dict]:
        query: dict[str, Any] = {}
        since_oid = parse_object_id(since_id)
        if since_oid is not None:
            query["_id"] = {"$gt": since_oid}

        user_oid = parse_object_id(user_id)
        if user_id is not None:
            query["user_id"] = user_oid

        rows = list(self.threat_alerts.find(query).sort("_id", ASCENDING).limit(limit))
        return [serialize_document(row) for row in rows]

    def cleanup_old_records(self, retention_days: int = 90) -> dict[str, int]:
        cutoff = utcnow() - dt.timedelta(days=max(7, int(retention_days)))
        scan_result = self.scan_records.delete_many({"created_at": {"$lt": cutoff}})
        alert_result = self.threat_alerts.delete_many({"created_at": {"$lt": cutoff}})
        queue_result = self.active_learning_queue.delete_many({"created_at": {"$lt": cutoff}, "status": {"$ne": "pending"}})
        cloud_result = self.cloud_verification_queue.delete_many({"created_at": {"$lt": cutoff}, "status": {"$in": ["failed", "verified"]}})

        return {
            "scan_records_deleted": int(scan_result.deleted_count),
            "alerts_deleted": int(alert_result.deleted_count),
            "active_learning_deleted": int(queue_result.deleted_count),
            "cloud_queue_deleted": int(cloud_result.deleted_count),
        }
