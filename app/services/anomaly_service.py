from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services.feature_extractors import build_url_feature_dict


class AnomalyService:
    def __init__(self, model_dir: str | Path, fraud_iso_threshold: float = -0.02):
        self.model_dir = Path(model_dir)
        self.fraud_iso_threshold = float(fraud_iso_threshold)
        self._detectors: dict[str, Any] = {}

    def _load_optional_detector(self, key: str, candidates: list[str]):
        if key in self._detectors:
            return self._detectors[key]

        detector = None
        for filename in candidates:
            path = self.model_dir / filename
            if not path.exists():
                continue
            try:
                detector = joblib.load(path)
                break
            except Exception:
                detector = None
        self._detectors[key] = detector
        return detector

    @staticmethod
    def _entropy(value: str | bytes) -> float:
        if not value:
            return 0.0
        if isinstance(value, str):
            data = value.encode("utf-8", errors="ignore")
        else:
            data = bytes(value)
        if not data:
            return 0.0

        counts: dict[int, int] = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1
        entropy = 0.0
        total = len(data)
        for count in counts.values():
            prob = count / total
            entropy -= prob * (math.log(prob) / math.log(2.0))
        return float(entropy)

    @staticmethod
    def _score_with_detector(detector, frame: pd.DataFrame) -> dict | None:
        if detector is None:
            return None

        try:
            feature_names = list(getattr(detector, "feature_names_in_", []))
            working = frame.copy()
            if feature_names:
                for column in feature_names:
                    if column not in working.columns:
                        working[column] = 0.0
                working = working[feature_names]

            prediction = None
            if hasattr(detector, "predict"):
                prediction = int(detector.predict(working)[0])

            score = None
            if hasattr(detector, "decision_function"):
                score = float(detector.decision_function(working)[0])
            elif hasattr(detector, "score_samples"):
                score = float(detector.score_samples(working)[0])

            flagged = prediction == -1
            if prediction is None and score is not None:
                flagged = score < 0

            return {
                "flagged": bool(flagged),
                "score": score,
                "source": detector.__class__.__name__,
            }
        except Exception:
            return None

    @staticmethod
    def _text_features(text: str) -> dict[str, float]:
        value = str(text or "")
        length = len(value)
        lowered = value.lower()
        tokens = re.findall(r"[a-zA-Z0-9]+", lowered)
        unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
        links = len(re.findall(r"https?://|www\.", lowered))

        return {
            "length": float(length),
            "digit_ratio": (sum(char.isdigit() for char in value) / length) if length else 0.0,
            "uppercase_ratio": (sum(char.isupper() for char in value) / length) if length else 0.0,
            "punctuation_ratio": (sum(not char.isalnum() and not char.isspace() for char in value) / length) if length else 0.0,
            "link_count": float(links),
            "entropy": AnomalyService._entropy(value),
            "unique_token_ratio": float(unique_ratio),
        }

    def assess_url(self, url: str) -> dict:
        feature_map = build_url_feature_dict(url)
        frame = pd.DataFrame([feature_map])
        detector = self._load_optional_detector(
            "url",
            ["url_isolation_forest.pkl", "url_oneclass_svm.pkl"],
        )
        model_signal = self._score_with_detector(detector, frame)
        if model_signal is not None:
            return model_signal

        heuristic_flag = feature_map["entropy"] >= 4.4 and feature_map["url_length"] >= 160
        heuristic_flag = heuristic_flag or bool(feature_map["has_obfuscated_percent"])
        return {
            "flagged": bool(heuristic_flag),
            "score": float(feature_map["entropy"]),
            "source": "heuristic_url_entropy",
        }

    def assess_email(self, subject: str, message: str) -> dict:
        composed = f"{subject or ''}\n{message or ''}".strip()
        features = self._text_features(composed)
        frame = pd.DataFrame([features])
        detector = self._load_optional_detector(
            "email",
            ["email_isolation_forest.pkl", "email_oneclass_svm.pkl"],
        )
        model_signal = self._score_with_detector(detector, frame)
        if model_signal is not None:
            return model_signal

        flagged = features["link_count"] >= 2 and features["entropy"] > 4.5 and features["uppercase_ratio"] > 0.2
        return {
            "flagged": bool(flagged),
            "score": float(features["entropy"]),
            "source": "heuristic_email_text",
        }

    def assess_message(self, message: str) -> dict:
        features = self._text_features(message)
        frame = pd.DataFrame([features])
        detector = self._load_optional_detector(
            "message",
            ["message_isolation_forest.pkl", "message_oneclass_svm.pkl"],
        )
        model_signal = self._score_with_detector(detector, frame)
        if model_signal is not None:
            return model_signal

        flagged = features["link_count"] >= 1 and features["entropy"] > 4.3 and features["length"] > 180
        return {
            "flagged": bool(flagged),
            "score": float(features["entropy"]),
            "source": "heuristic_message_text",
        }

    def assess_file(self, file_bytes: bytes, filename: str = "", pe_features: dict | None = None) -> dict:
        value = file_bytes or b""
        sample = value[:4096]
        entropy = self._entropy(sample)
        extension = Path(str(filename or "")).suffix.lower()
        size_bytes = len(value)
        feature_row = {
            "size_bytes": float(size_bytes),
            "header_entropy": float(entropy),
            "is_executable_like": 1.0 if extension in {".exe", ".dll", ".scr", ".js", ".vbs", ".ps1"} else 0.0,
        }
        if pe_features:
            for key, val in pe_features.items():
                if isinstance(val, (float, int, np.floating, np.integer)):
                    feature_row[key] = float(val)

        frame = pd.DataFrame([feature_row])
        detector = self._load_optional_detector(
            "file",
            ["file_isolation_forest.pkl", "file_oneclass_svm.pkl"],
        )
        model_signal = self._score_with_detector(detector, frame)
        if model_signal is not None:
            return model_signal

        flagged = entropy > 7.2 and size_bytes < 250000
        flagged = flagged or extension in {".js", ".vbs", ".ps1"}
        return {
            "flagged": bool(flagged),
            "score": float(entropy),
            "source": "heuristic_file_entropy",
        }

    def assess_fraud(self, prediction_details: dict | None = None) -> dict:
        details = prediction_details or {}
        iso_score = details.get("iso_anomaly_score")
        if iso_score is None:
            return {
                "flagged": False,
                "score": None,
                "source": "fraud_iso_missing",
            }

        score = float(iso_score)
        flagged = score < self.fraud_iso_threshold
        return {
            "flagged": bool(flagged),
            "score": score,
            "source": "fraud_isolation_forest",
        }
