import logging
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from app.services.feature_extractors import (
    build_fraud_feature_frame,
    build_url_feature_dict,
    build_url_feature_frame,
    extract_pe_aggregate_features,
)

LOGGER = logging.getLogger(__name__)

EMAIL_FEATURE_KEYWORDS = (
    "urgent",
    "verify",
    "account",
    "password",
    "wire transfer",
    "bank",
    "claim prize",
    "won",
    "bitcoin",
    "gift card",
    "suspended",
    "login",
    "invoice",
    "payment",
)

URL_SHORTENER_HINTS = (
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "adf.ly",
)

UCI_URL_FEATURE_COLUMNS = {
    "Index",
    "UsingIP",
    "LongURL",
    "ShortURL",
    "Symbol@",
    "Redirecting//",
    "PrefixSuffix-",
    "SubDomains",
    "HTTPS",
    "DomainRegLen",
    "Favicon",
    "NonStdPort",
    "HTTPSDomainURL",
    "RequestURL",
    "AnchorURL",
    "LinksInScriptTags",
    "ServerFormHandler",
    "InfoEmail",
    "AbnormalURL",
    "WebsiteForwarding",
    "StatusBarCust",
    "DisableRightClick",
    "UsingPopupWindow",
    "IframeRedirection",
    "AgeofDomain",
    "DNSRecording",
    "WebsiteTraffic",
    "PageRank",
    "GoogleIndex",
    "LinksPointingToPage",
    "StatsReport",
}


class ModelServiceError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

        self._email_tokenizer = None
        self._email_model = None
        self._torch = None
        self._email_preprocessor = None
        self._email_pickle_model = None
        self._email_model_name = None
        self._email_threshold = 0.5
        self._email_aux_vectorizer = None
        self._email_aux_model = None

        self._message_vectorizer = None
        self._message_model = None

        self._url_model = None
        self._url_model_kind = None
        self._url_label_encoder = None
        self._url_feature_names = None
        self._url_model_name = None
        self._url_aux_vectorizer = None
        self._url_aux_model = None

        self._file_xgb = None
        self._file_rf = None
        self._file_feature_cols = None

        self._fraud_xgb = None
        self._fraud_iso = None
        self._fraud_scaler = None
        self._fraud_ohe = None

    @staticmethod
    def _ensure_pickle_compat() -> None:
        try:
            import numpy.core as numpy_core

            sys.modules.setdefault("numpy._core", numpy_core)
        except Exception:
            pass

        # Compatibility aliases for pickles trained from notebook "__main__" classes.
        try:
            import __main__

            if not hasattr(__main__, "TextPreprocessor"):
                class TextPreprocessor:  # pragma: no cover - compatibility shim
                    pass

                setattr(__main__, "TextPreprocessor", TextPreprocessor)

            if not hasattr(__main__, "PhishingEmailDetector"):
                class PhishingEmailDetector:  # pragma: no cover - compatibility shim
                    pass

                setattr(__main__, "PhishingEmailDetector", PhishingEmailDetector)
        except Exception:
            pass

    def _safe_joblib_load(self, path: Path, optional: bool = False):
        self._ensure_pickle_compat()
        # These pickles were trained with numpy>=2 and scikit-learn>=1.6.
        # Guarding here prevents low-level interpreter crashes on incompatible runtimes.
        try:
            import sklearn

            sklearn_parts = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
            numpy_major = int(np.__version__.split(".")[0])
            if sklearn_parts < (1, 6) or numpy_major < 2:
                if optional:
                    return None
                raise ModelServiceError(
                    "Incompatible runtime for model artifacts. Install dependencies from app/requirements.txt "
                    "(numpy>=2 and scikit-learn>=1.6)."
                )
        except ModelServiceError:
            raise
        except Exception:
            # If version parsing fails for any reason, continue and attempt load.
            pass

        try:
            return joblib.load(path)
        except Exception as exc:
            if optional:
                LOGGER.warning("Optional artifact failed to load (%s): %s", path.name, exc)
                return None

            raise ModelServiceError(
                f"Failed to load '{path.name}'. This usually means a dependency/version mismatch. "
                "Install dependencies from app/requirements.txt."
            ) from exc

    def _resolve_model_path(self, candidates: list[str]) -> Path:
        for candidate in candidates:
            path = self.model_dir / candidate
            if path.exists():
                return path
        raise ModelServiceError(f"None of the model files were found: {candidates}")

    def _load_email_bundle(self) -> None:
        if self._email_pickle_model is not None and self._email_preprocessor is not None:
            return
        if self._email_model is not None and self._email_tokenizer is not None:
            return

        detector_path = self.model_dir / "email_detector_model.pkl"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        if detector_path.exists() and preprocessor_path.exists():
            loaded_detector = self._safe_joblib_load(detector_path)
            loaded_preprocessor = self._safe_joblib_load(preprocessor_path)

            model_candidate = loaded_detector
            if hasattr(loaded_detector, "model"):
                model_candidate = getattr(loaded_detector, "model")
                if loaded_preprocessor is None and hasattr(loaded_detector, "preprocessor"):
                    loaded_preprocessor = getattr(loaded_detector, "preprocessor")

            if not any(
                hasattr(model_candidate, method_name)
                for method_name in ("predict_proba", "decision_function", "predict")
            ):
                raise ModelServiceError(
                    "email_detector_model.pkl does not expose a compatible sklearn prediction interface."
                )

            if not hasattr(loaded_preprocessor, "transform"):
                raise ModelServiceError(
                    "preprocessor.pkl must provide a transform(...) method for email feature preprocessing."
                )

            threshold = getattr(loaded_detector, "threshold", 0.5)
            try:
                threshold = float(threshold)
            except Exception:
                threshold = 0.5

            self._email_pickle_model = model_candidate
            self._email_preprocessor = loaded_preprocessor
            self._email_model_name = detector_path.name
            self._email_threshold = max(0.0, min(1.0, threshold))
            return

        model_path = self._resolve_model_path(["email_spam_distilbert_model", "email_spam_model"])

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()

            self._email_tokenizer = tokenizer
            self._email_model = model
            self._torch = torch
            self._email_model_name = str(model_path.name)
        except Exception as exc:
            LOGGER.warning("Email model loading failed, heuristic fallback will be used: %s", exc)
            self._email_model = False
            self._email_tokenizer = False
            self._email_model_name = "heuristic_fallback"

        aux_vectorizer = self.model_dir / "email_char_vectorizer.pkl"
        aux_model = self.model_dir / "email_char_model.pkl"
        if aux_vectorizer.exists() and aux_model.exists():
            self._email_aux_vectorizer = self._safe_joblib_load(aux_vectorizer, optional=True)
            self._email_aux_model = self._safe_joblib_load(aux_model, optional=True)

    def _load_message_bundle(self) -> None:
        if self._message_model is not None and self._message_vectorizer is not None:
            return

        model_path = self.model_dir / "sms_rf_tfidf_model.pkl"
        vectorizer_path = self.model_dir / "sms_tfidf_vectorizer.pkl"
        if not model_path.exists() or not vectorizer_path.exists():
            raise ModelServiceError("Message model artifacts are missing from app/models")

        self._message_model = self._safe_joblib_load(model_path)
        self._message_vectorizer = self._safe_joblib_load(vectorizer_path)

    def _load_url_bundle(self) -> None:
        if self._url_model is not None:
            return

        preferred_pickle = self.model_dir / "url_model.pkl"
        if preferred_pickle.exists():
            model = self._safe_joblib_load(preferred_pickle)
            self._url_model = model
            self._url_model_kind = "pickle"
            self._url_model_name = preferred_pickle.name
            feature_names = list(getattr(model, "feature_names_in_", []))
            self._url_feature_names = feature_names if feature_names else list(build_url_feature_dict("").keys())

            encoder_path = self.model_dir / "label_encoder.joblib"
            if encoder_path.exists():
                self._url_label_encoder = self._safe_joblib_load(encoder_path, optional=True)
            return

        import xgboost as xgb

        model_path = self._resolve_model_path([
            "url_xgboost_improved.json",
            "url_xgboost_malicious_detector.json",
        ])

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        booster = model.get_booster()

        feature_names = booster.feature_names
        if not feature_names:
            feature_names = [f"f{index}" for index in range(model.n_features_in_)]

        self._url_model = model
        self._url_model_kind = "xgb_json"
        self._url_feature_names = feature_names
        self._url_model_name = model_path.name

        encoder_path = self.model_dir / "label_encoder.joblib"
        if encoder_path.exists():
            self._url_label_encoder = self._safe_joblib_load(encoder_path, optional=True)

        aux_vectorizer_path = self.model_dir / "url_char_vectorizer.pkl"
        aux_model_path = self.model_dir / "url_char_model.pkl"
        if aux_vectorizer_path.exists() and aux_model_path.exists():
            self._url_aux_vectorizer = self._safe_joblib_load(aux_vectorizer_path, optional=True)
            self._url_aux_model = self._safe_joblib_load(aux_model_path, optional=True)

    def _load_file_bundle(self) -> None:
        if self._file_xgb is not None and self._file_rf is not None and self._file_feature_cols is not None:
            return

        import xgboost as xgb

        xgb_path = self.model_dir / "file_malware_xgboost.json"
        rf_path = self.model_dir / "file_malware_rf.pkl"
        cols_path = self.model_dir / "feature_cols.pkl"

        if not xgb_path.exists() or not rf_path.exists() or not cols_path.exists():
            raise ModelServiceError("File malware model artifacts are missing from app/models")

        file_xgb = xgb.XGBClassifier()
        file_xgb.load_model(str(xgb_path))

        self._file_xgb = file_xgb
        self._file_rf = self._safe_joblib_load(rf_path)
        self._file_feature_cols = self._safe_joblib_load(cols_path)

    def _load_fraud_bundle(self) -> None:
        if (
            self._fraud_xgb is not None
            and self._fraud_iso is not None
            and self._fraud_scaler is not None
            and self._fraud_ohe is not None
        ):
            return

        import xgboost as xgb

        xgb_path = self.model_dir / "fraud_xgboost.json"
        iso_path = self.model_dir / "fraud_iso_forest.pkl"
        scaler_path = self.model_dir / "fraud_scaler.pkl"
        ohe_path = self.model_dir / "fraud_ohe.pkl"

        if not all(path.exists() for path in [xgb_path, iso_path, scaler_path, ohe_path]):
            raise ModelServiceError("Fraud model artifacts are missing from app/models")

        fraud_xgb = xgb.XGBClassifier()
        fraud_xgb.load_model(str(xgb_path))

        self._fraud_xgb = fraud_xgb
        self._fraud_iso = self._safe_joblib_load(iso_path)
        self._fraud_scaler = self._safe_joblib_load(scaler_path)
        self._fraud_ohe = self._safe_joblib_load(ohe_path)

    @staticmethod
    def _heuristic_email_score(text: str) -> float:
        flags = [
            "urgent",
            "verify",
            "account",
            "password",
            "wire transfer",
            "bank",
            "claim prize",
            "won",
            "bitcoin",
            "gift card",
            "suspended",
            "login now",
        ]
        lowered = text.lower()
        hits = sum(1 for keyword in flags if keyword in lowered)
        return min(0.95, 0.18 * hits)

    @staticmethod
    def _blend_probabilities(primary: float, secondary: float | None = None, secondary_weight: float = 0.35) -> float:
        primary_value = float(primary)
        if secondary is None:
            return primary_value

        aux_weight = max(0.0, min(1.0, float(secondary_weight)))
        return ((1.0 - aux_weight) * primary_value) + (aux_weight * float(secondary))

    @staticmethod
    def _clean_email_text(value: str) -> str:
        lowered = str(value or "").lower()
        lowered = re.sub(r"[^a-z0-9\s@._$%!?/:+-]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    @classmethod
    def _build_email_feature_frame(cls, text: str) -> pd.DataFrame:
        raw_text = str(text or "")
        processed = cls._clean_email_text(raw_text)
        words = [token for token in re.split(r"\s+", processed) if token]

        keyword_count = sum(processed.count(keyword) for keyword in EMAIL_FEATURE_KEYWORDS)
        word_count = len(words)
        avg_word_length = (sum(len(word) for word in words) / word_count) if word_count else 0.0
        url_count = len(re.findall(r"(?:https?://|www\.)\S+", raw_text, flags=re.IGNORECASE))
        email_count = len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", raw_text))

        row = {
            "processed_text": processed,
            "keyword_count": float(keyword_count),
            "text_length": float(len(raw_text)),
            "word_count": float(word_count),
            "avg_word_length": float(avg_word_length),
            "url_count": float(url_count),
            "email_count": float(email_count),
            "exclamation_count": float(raw_text.count("!")),
            "question_count": float(raw_text.count("?")),
            "dollar_count": float(raw_text.count("$")),
        }
        return pd.DataFrame([row])

    @staticmethod
    def _looks_like_uci_url_schema(columns: list[str] | None) -> bool:
        if not columns:
            return False
        return set(columns).issubset(UCI_URL_FEATURE_COLUMNS)

    @staticmethod
    def _to_uci_signal(value: bool) -> float:
        return -1.0 if value else 1.0

    @classmethod
    def _build_uci_url_feature_dict(cls, url: str) -> dict[str, float]:
        value = str(url or "").strip()
        if "://" not in value:
            value = f"http://{value}"

        try:
            parsed = urlparse(value)
            domain = (parsed.hostname or "").lower()
            path = parsed.path or ""
            query = parsed.query or ""
            port = parsed.port
        except Exception:
            domain = ""
            path = ""
            query = ""
            port = None

        tail = value.split("://", 1)[-1]
        tail_lower = tail.lower()
        dots = domain.count(".")
        url_length = len(value)
        scheme = (urlparse(value).scheme or "").lower()

        using_ip = bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", domain))
        has_shortener = any(shortener in tail_lower for shortener in URL_SHORTENER_HINTS)
        has_redirect = "//" in tail

        if url_length < 54:
            long_url = 1.0
        elif url_length <= 75:
            long_url = 0.0
        else:
            long_url = -1.0

        if dots <= 1:
            subdomains = 1.0
        elif dots == 2:
            subdomains = 0.0
        else:
            subdomains = -1.0

        feature_map = {
            "Index": 0.0,
            "UsingIP": cls._to_uci_signal(using_ip),
            "LongURL": long_url,
            "ShortURL": cls._to_uci_signal(has_shortener),
            "Symbol@": cls._to_uci_signal("@" in value),
            "Redirecting//": cls._to_uci_signal(has_redirect),
            "PrefixSuffix-": cls._to_uci_signal("-" in domain),
            "SubDomains": subdomains,
            "HTTPS": 1.0 if scheme == "https" else -1.0,
            "DomainRegLen": 0.0,
            "Favicon": 0.0,
            "NonStdPort": cls._to_uci_signal(port not in {None, 80, 443}),
            "HTTPSDomainURL": cls._to_uci_signal(("https" in domain) and (scheme != "https")),
            "RequestURL": 0.0,
            "AnchorURL": cls._to_uci_signal("javascript:" in tail_lower),
            "LinksInScriptTags": 0.0,
            "ServerFormHandler": cls._to_uci_signal("mailto:" in tail_lower),
            "InfoEmail": cls._to_uci_signal(("mailto:" in tail_lower) or ("@" in query)),
            "AbnormalURL": cls._to_uci_signal(bool(domain and domain not in tail_lower)),
            "WebsiteForwarding": cls._to_uci_signal(tail.count("//") > 0),
            "StatusBarCust": cls._to_uci_signal("status=" in tail_lower),
            "DisableRightClick": 0.0,
            "UsingPopupWindow": cls._to_uci_signal("popup" in tail_lower),
            "IframeRedirection": cls._to_uci_signal("iframe" in tail_lower),
            "AgeofDomain": 0.0,
            "DNSRecording": 0.0,
            "WebsiteTraffic": 0.0,
            "PageRank": 0.0,
            "GoogleIndex": 0.0,
            "LinksPointingToPage": float(path.count("/") + query.count("&")),
            "StatsReport": cls._to_uci_signal(any(keyword in domain for keyword in ("-secure", "login", "verify"))),
        }
        return feature_map

    @staticmethod
    def _normalize_label(value: Any) -> str:
        return str(value).strip().lower()

    @classmethod
    def _url_malicious_probability(cls, probabilities: np.ndarray, labels: list[Any]) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        normalized_labels = [cls._normalize_label(label) for label in labels]

        if normalized_labels and len(normalized_labels) == len(probs):
            if set(normalized_labels) == {"-1", "1"}:
                malicious = float(
                    sum(prob for prob, label in zip(probs, normalized_labels) if label == "-1")
                )
                return max(0.0, min(1.0, malicious))

            malicious_aliases = {
                "1",
                "-1",
                "malicious",
                "phishing",
                "malware",
                "defacement",
                "scam",
                "spam",
                "fraud",
                "threat",
                "attack",
                "suspicious",
            }
            benign_aliases = {
                "0",
                "benign",
                "safe",
                "legit",
                "legitimate",
                "clean",
                "allow",
            }

            malicious_indexes = [index for index, label in enumerate(normalized_labels) if label in malicious_aliases]
            if malicious_indexes:
                return max(0.0, min(1.0, float(sum(probs[index] for index in malicious_indexes))))

            benign_indexes = [index for index, label in enumerate(normalized_labels) if label in benign_aliases]
            if benign_indexes:
                benign_prob = float(sum(probs[index] for index in benign_indexes))
                return max(0.0, min(1.0, 1.0 - benign_prob))

        if len(probs) == 2:
            return float(probs[1])
        return float(max(probs))

    @classmethod
    def _format_url_label(cls, label: Any, labels: list[Any]) -> str:
        normalized = cls._normalize_label(label)
        normalized_labels = {cls._normalize_label(item) for item in labels}
        if normalized_labels == {"-1", "1"}:
            return "malicious" if normalized == "-1" else "benign"
        return str(label)

    @staticmethod
    def _is_positive_class(value: Any) -> bool:
        text = str(value).strip().lower()
        return text in {"1", "spam", "malicious", "fraud", "phishing", "scam", "true", "positive"}

    @staticmethod
    def _is_benign_class(value: Any) -> bool:
        text = str(value).strip().lower()
        return text in {"0", "ham", "safe", "benign", "legit", "legitimate", "clean", "false", "negative"}

    @classmethod
    def _extract_positive_probability(cls, probabilities: np.ndarray, classes: list[Any] | None = None) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        labels = list(classes or [])

        if labels and len(labels) == len(probs):
            positive_indexes = [index for index, label in enumerate(labels) if cls._is_positive_class(label)]
            if positive_indexes:
                return float(sum(probs[index] for index in positive_indexes))

            benign_indexes = [index for index, label in enumerate(labels) if cls._is_benign_class(label)]
            if benign_indexes:
                benign_prob = float(sum(probs[index] for index in benign_indexes))
                return float(max(0.0, min(1.0, 1.0 - benign_prob)))

        if len(probs) == 2:
            return float(probs[1])
        return float(max(probs))

    @staticmethod
    def _predict_proba_with_fallback(model: Any, *payloads: Any) -> np.ndarray:
        last_error = None
        for payload in payloads:
            if payload is None:
                continue
            try:
                probabilities = model.predict_proba(payload)
                values = np.asarray(probabilities, dtype=float)
                if values.ndim == 1:
                    return values
                if values.ndim == 2 and values.shape[0] >= 1:
                    return values[0]
            except Exception as exc:
                last_error = exc

        if hasattr(model, "decision_function"):
            for payload in payloads:
                if payload is None:
                    continue
                try:
                    raw_score = model.decision_function(payload)
                    score_value = float(np.asarray(raw_score).reshape(-1)[0])
                    positive = 1.0 / (1.0 + np.exp(-score_value))
                    return np.asarray([1.0 - positive, positive], dtype=float)
                except Exception as exc:
                    last_error = exc

        if hasattr(model, "predict"):
            for payload in payloads:
                if payload is None:
                    continue
                try:
                    predicted = model.predict(payload)
                    value = str(np.asarray(predicted).reshape(-1)[0]).strip().lower()
                    positive = 1.0 if value in {"1", "spam", "malicious", "fraud", "phishing", "scam"} else 0.0
                    return np.asarray([1.0 - positive, positive], dtype=float)
                except Exception as exc:
                    last_error = exc

        raise ModelServiceError("The loaded model does not expose a compatible prediction interface.") from last_error

    def predict_email(self, subject: str, message: str) -> dict:
        text = f"{subject or ''}\n{message or ''}".strip()
        if not text:
            raise ModelServiceError("Email scan requires at least subject or message text")

        self._load_email_bundle()

        decision_threshold = 0.5
        if self._email_pickle_model is not None and self._email_preprocessor is not None:
            try:
                feature_frame = self._build_email_feature_frame(text)
                transformed = self._email_preprocessor.transform(feature_frame)
                probabilities = self._predict_proba_with_fallback(
                    self._email_pickle_model,
                    transformed,
                    feature_frame,
                )
                classes = list(getattr(self._email_pickle_model, "classes_", []))
                spam_prob = self._extract_positive_probability(probabilities, classes=classes)
                decision_threshold = float(self._email_threshold or 0.5)
                source = self._email_model_name or "email_detector_model.pkl"
            except Exception as exc:
                LOGGER.warning("Email pickle model inference fallback to heuristic: %s", exc)
                spam_prob = self._heuristic_email_score(text)
                source = "heuristic_fallback"
        elif self._email_model is False:
            spam_prob = self._heuristic_email_score(text)
            source = "heuristic"
        else:
            try:
                inputs = self._email_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                # DistilBERT does not use token_type_ids.
                inputs.pop("token_type_ids", None)

                with self._torch.no_grad():
                    logits = self._email_model(**inputs).logits
                    probs = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()
                spam_prob = float(probs[1] if len(probs) > 1 else probs[0])
                source = self._email_model_name or "distilbert"
            except Exception as exc:
                LOGGER.warning("Email inference fallback to heuristic: %s", exc)
                spam_prob = self._heuristic_email_score(text)
                source = "heuristic_fallback"

        auxiliary_prob = None
        using_legacy_email_stack = self._email_pickle_model is None
        if using_legacy_email_stack and self._email_aux_vectorizer is not None and self._email_aux_model is not None:
            try:
                aux_transformed = self._email_aux_vectorizer.transform([text])
                auxiliary_prob = float(self._email_aux_model.predict_proba(aux_transformed)[0][1])
                spam_prob = self._blend_probabilities(spam_prob, auxiliary_prob, secondary_weight=0.35)
                source = f"{source}+char_model"
            except Exception as exc:
                LOGGER.warning("Email auxiliary model failed: %s", exc)

        verdict = "SPAM" if spam_prob >= decision_threshold else "HAM"
        confidence = spam_prob if verdict == "SPAM" else 1.0 - spam_prob

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "risk_score": round(float(spam_prob * 100.0), 2),
            "details": {
                "spam_probability": round(float(spam_prob), 6),
                "auxiliary_spam_probability": round(float(auxiliary_prob), 6) if auxiliary_prob is not None else None,
                "decision_threshold": round(float(decision_threshold), 4),
                "source": source,
            },
        }

    def predict_message(self, message_text: str) -> dict:
        if not str(message_text or "").strip():
            raise ModelServiceError("Message scan requires non-empty text")

        self._load_message_bundle()

        transformed = self._message_vectorizer.transform([message_text])
        prediction = int(self._message_model.predict(transformed)[0])

        probabilities = self._message_model.predict_proba(transformed)[0]
        classes = list(getattr(self._message_model, "classes_", [0, 1]))
        class_to_prob = {int(cls): float(probabilities[index]) for index, cls in enumerate(classes)}

        scam_probability = class_to_prob.get(1, float(max(probabilities)))
        verdict = "SCAM" if prediction == 1 else "SAFE"
        confidence = scam_probability if verdict == "SCAM" else 1.0 - scam_probability

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "risk_score": round(float(scam_probability * 100.0), 2),
            "details": {
                "scam_probability": round(float(scam_probability), 6),
                "class_probabilities": {
                    "ham": round(class_to_prob.get(0, 0.0), 6),
                    "spam": round(class_to_prob.get(1, 0.0), 6),
                },
                "model": "sms_rf_tfidf_model",
            },
        }

    def predict_url(self, url: str) -> dict:
        if not str(url or "").strip():
            raise ModelServiceError("URL scan requires a URL string")

        self._load_url_bundle()

        default_features = build_url_feature_dict(url)
        expected_columns = self._url_feature_names if self._url_feature_names else list(default_features.keys())
        if self._url_model_kind == "pickle" and self._looks_like_uci_url_schema(expected_columns):
            uci_features = self._build_uci_url_feature_dict(url)
            row = {column: float(uci_features.get(column, 0.0)) for column in expected_columns}
            feature_frame = pd.DataFrame([row], columns=expected_columns)
        else:
            feature_frame = build_url_feature_frame(url, expected_columns)

        probabilities = self._predict_proba_with_fallback(
            self._url_model,
            feature_frame,
            pd.DataFrame([default_features]),
            [url],
        )
        predicted_index = int(np.argmax(probabilities))

        model_classes = list(getattr(self._url_model, "classes_", []))
        if model_classes and len(model_classes) == len(probabilities):
            labels = model_classes
        elif self._url_label_encoder is not None:
            encoder_classes = list(getattr(self._url_label_encoder, "classes_", []))
            if len(encoder_classes) == len(probabilities):
                labels = encoder_classes
            else:
                labels = model_classes if model_classes else [f"class_{index}" for index in range(len(probabilities))]
        elif len(probabilities) == 4:
            labels = ["benign", "defacement", "malware", "phishing"]
        else:
            labels = [f"class_{index}" for index in range(len(probabilities))]

        class_probabilities: dict[str, float] = {}
        for index in range(min(len(labels), len(probabilities))):
            label_name = self._format_url_label(labels[index], labels)
            class_probabilities[label_name] = round(float(probabilities[index]), 6)

        predicted_label = labels[predicted_index] if predicted_index < len(labels) else str(predicted_index)
        base_malicious_probability = self._url_malicious_probability(probabilities, labels)

        aux_malicious_probability = None
        using_legacy_url_stack = self._url_model_kind != "pickle"
        if using_legacy_url_stack and self._url_aux_vectorizer is not None and self._url_aux_model is not None:
            try:
                aux_transformed = self._url_aux_vectorizer.transform([url])
                aux_probs = self._url_aux_model.predict_proba(aux_transformed)[0]
                if len(aux_probs) >= 2:
                    aux_malicious_probability = float(aux_probs[1])
                else:
                    aux_malicious_probability = float(aux_probs[0])
            except Exception as exc:
                LOGGER.warning("URL auxiliary model failed: %s", exc)

        malicious_probability = self._blend_probabilities(
            base_malicious_probability,
            aux_malicious_probability,
            secondary_weight=0.35,
        )

        verdict = "MALICIOUS" if malicious_probability >= 0.5 else "SAFE"
        confidence = malicious_probability if verdict == "MALICIOUS" else 1.0 - malicious_probability

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "risk_score": round(float(malicious_probability * 100.0), 2),
            "details": {
                "predicted_category": self._format_url_label(predicted_label, labels),
                "class_probabilities": class_probabilities,
                "base_malicious_probability": round(float(base_malicious_probability), 6),
                "auxiliary_malicious_probability": round(float(aux_malicious_probability), 6)
                if aux_malicious_probability is not None
                else None,
                "model": self._url_model_name or "url_model",
            },
        }

    def predict_file(self, file_bytes: bytes, filename: str = "") -> dict:
        if not file_bytes:
            raise ModelServiceError("File scan requires non-empty binary content")

        self._load_file_bundle()
        feature_map = extract_pe_aggregate_features(file_bytes)

        if feature_map is None:
            return {
                "verdict": "NOT_PE_OR_INVALID",
                "confidence": 1.0,
                "risk_score": 10.0,
                "details": {
                    "reason": "The uploaded file is not a valid PE executable or could not be parsed.",
                    "filename": filename,
                },
            }

        frame = pd.DataFrame([{column: feature_map.get(column, 0.0) for column in self._file_feature_cols}])
        xgb_prob = float(self._file_xgb.predict_proba(frame)[0][1])
        rf_prob = float(self._file_rf.predict_proba(frame)[0][1])
        ensemble_prob = (xgb_prob + rf_prob) / 2.0

        verdict = "MALWARE" if ensemble_prob >= 0.5 else "GOODWARE"
        confidence = ensemble_prob if verdict == "MALWARE" else 1.0 - ensemble_prob

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "risk_score": round(float(ensemble_prob * 100.0), 2),
            "details": {
                "xgboost_probability": round(xgb_prob, 6),
                "random_forest_probability": round(rf_prob, 6),
                "filename": filename,
                "pe_aggregate_features": {key: round(float(value), 6) for key, value in feature_map.items()},
            },
        }

    def predict_fraud(self, transaction: dict) -> dict:
        if not isinstance(transaction, dict):
            raise ModelServiceError("Fraud scan requires a JSON object payload")

        self._load_fraud_bundle()

        frame = build_fraud_feature_frame(transaction, self._fraud_ohe, self._fraud_scaler)

        iso_columns = list(getattr(self._fraud_iso, "feature_names_in_", []))
        if iso_columns:
            for column in iso_columns:
                if column not in frame.columns:
                    frame[column] = 0.0
            iso_input = frame[iso_columns]
        else:
            iso_input = frame

        iso_score = float(self._fraud_iso.decision_function(iso_input)[0])
        frame["iso_anomaly_score"] = iso_score

        xgb_columns = self._fraud_xgb.get_booster().feature_names
        if xgb_columns:
            for column in xgb_columns:
                if column not in frame.columns:
                    frame[column] = 0.0
            model_input = frame[xgb_columns]
        else:
            model_input = frame

        fraud_probability = float(self._fraud_xgb.predict_proba(model_input)[0][1])
        verdict = "FRAUD" if fraud_probability >= 0.5 else "LEGIT"
        confidence = fraud_probability if verdict == "FRAUD" else 1.0 - fraud_probability

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "risk_score": round(float(fraud_probability * 100.0), 2),
            "details": {
                "fraud_probability": round(fraud_probability, 6),
                "iso_anomaly_score": round(iso_score, 6),
                "model": "fraud_xgboost + isolation_forest",
            },
        }

    def get_status(self) -> dict:
        email_pickle_ready = all(
            (self.model_dir / file_name).exists()
            for file_name in ["email_detector_model.pkl", "preprocessor.pkl"]
        )
        email_legacy_ready = any(
            (self.model_dir / file_name).exists()
            for file_name in ["email_spam_distilbert_model", "email_spam_model"]
        )

        url_pickle_ready = (self.model_dir / "url_model.pkl").exists()
        url_legacy_ready = any(
            (self.model_dir / file_name).exists()
            for file_name in ["url_xgboost_improved.json", "url_xgboost_malicious_detector.json"]
        )

        status = {
            "email": email_pickle_ready or email_legacy_ready,
            "message": all(
                (self.model_dir / file_name).exists()
                for file_name in ["sms_rf_tfidf_model.pkl", "sms_tfidf_vectorizer.pkl"]
            ),
            "url": url_pickle_ready or url_legacy_ready,
            "file": all(
                (self.model_dir / file_name).exists()
                for file_name in ["file_malware_xgboost.json", "file_malware_rf.pkl", "feature_cols.pkl"]
            ),
            "fraud": all(
                (self.model_dir / file_name).exists()
                for file_name in ["fraud_xgboost.json", "fraud_iso_forest.pkl", "fraud_scaler.pkl", "fraud_ohe.pkl"]
            ),
        }

        status["email_model_profile"] = (
            "email_detector_model.pkl+preprocessor.pkl" if email_pickle_ready else "distilbert_legacy" if email_legacy_ready else "missing"
        )
        status["url_model_profile"] = (
            "url_model.pkl" if url_pickle_ready else "xgboost_json_legacy" if url_legacy_ready else "missing"
        )

        status["email_aux_model"] = all(
            (self.model_dir / file_name).exists()
            for file_name in ["email_char_vectorizer.pkl", "email_char_model.pkl"]
        )
        status["url_aux_model"] = all(
            (self.model_dir / file_name).exists()
            for file_name in ["url_char_vectorizer.pkl", "url_char_model.pkl"]
        )
        status["model_dir"] = str(self.model_dir)
        return status
