import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SUSPICIOUS_TLDS = [
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".pw",
    ".xyz",
    ".top",
    ".club",
    ".online",
    ".ru",
    ".cn",
]

SHORTENERS = [
    "bit.ly",
    "tinyurl",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "adf.ly",
]

SENSITIVE_WORDS = [
    "login",
    "signin",
    "bank",
    "secure",
    "account",
    "update",
    "verify",
    "password",
    "admin",
    "support",
    "confirm",
    "paypal",
    "ebay",
    "apple",
    "amazon",
]

BRAND_WORDS = [
    "paypal",
    "amazon",
    "apple",
    "google",
    "microsoft",
    "facebook",
    "netflix",
]


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    probs = [float(value.count(char)) / len(value) for char in dict.fromkeys(value)]
    return float(-sum(p * math.log(p) / math.log(2.0) for p in probs if p > 0))


def build_url_feature_dict(url: str) -> dict:
    value = str(url or "").strip()
    if "://" not in value:
        value = f"http://{value}"

    try:
        parsed = urlparse(value)
        domain = (parsed.netloc or "").lower()
        if "@" in domain:
            domain = domain.split("@")[-1]
        domain = domain.split(":")[0]
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        domain, path, query = "", "", ""

    full_url = value.lower()

    feature_map = {
        "url_length": len(value),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": value.count("."),
        "subdomain_count": max(0, domain.count(".") - 1),
        "has_ip": 1 if re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", domain) else 0,
        "has_at_symbol": 1 if "@" in value else 0,
        "has_redirection": 1 if "//" in value[8:] else 0,
        "is_https": 1 if value.lower().startswith("https") else 0,
        "has_shortener": 1 if any(shortener in full_url for shortener in SHORTENERS) else 0,
        "has_prefix_suffix_dash": 1 if re.search(r"^-|-$", domain) else 0,
        "depth": path.count("/"),
        "uppercase_count": sum(1 for char in value if char.isupper()),
        "uppercase_ratio": sum(1 for char in value if char.isupper()) / len(value) if value else 0.0,
        "digit_ratio": sum(1 for char in value if char.isdigit()) / len(value) if value else 0.0,
        "special_char_ratio": sum(1 for char in value if not char.isalnum()) / len(value) if value else 0.0,
        "hyphen_count": value.count("-"),
        "underscore_count": value.count("_"),
        "question_count": value.count("?"),
        "equal_count": value.count("="),
        "ampersand_count": value.count("&"),
        "percent_count": value.count("%"),
        "has_www": 1 if "www." in full_url else 0,
        "sensitive_word_count": sum(1 for word in SENSITIVE_WORDS if word in full_url),
        "brand_count": sum(1 for brand in BRAND_WORDS if brand in full_url),
        "entropy": shannon_entropy(value),
        "has_suspicious_tld": 1 if any(tld in domain for tld in SUSPICIOUS_TLDS) else 0,
        "num_digits_domain": sum(1 for char in domain if char.isdigit()),
        "vowel_ratio": sum(1 for char in domain if char in "aeiou") / len(domain) if domain else 0.0,
        "longest_word_length": max([len(word) for word in re.split(r"[\W_]+", value) if word], default=0),
        "has_punycode": 1 if "xn--" in domain else 0,
        "has_obfuscated_percent": 1 if value.count("%") > 5 else 0,
        "query_params_count": len(query.split("&")) if query else 0,
        "has_brand_in_subdomain": 1
        if domain and any(brand in domain.split(".")[0] for brand in ["paypal", "amazon", "apple"])
        else 0,
        "digit_in_domain_ratio": (sum(1 for char in domain if char.isdigit()) / len(domain)) if domain else 0.0,
    }

    return feature_map


def build_url_feature_frame(url: str, expected_columns: list[str]) -> pd.DataFrame:
    feature_map = build_url_feature_dict(url)
    row = {column: float(feature_map.get(column, 0.0)) for column in expected_columns}
    return pd.DataFrame([row], columns=expected_columns)


def extract_pe_aggregate_features(file_bytes: bytes) -> dict | None:
    try:
        import pefile
    except Exception:
        return None

    try:
        pe = pefile.PE(data=file_bytes, fast_load=True)
    except Exception:
        return None

    section_rows = []
    for section in pe.sections:
        try:
            section_rows.append(
                {
                    "size_of_data": float(section.SizeOfRawData),
                    "virtual_address": float(section.VirtualAddress),
                    "entropy": float(section.get_entropy()),
                    "virtual_size": float(section.Misc_VirtualSize),
                }
            )
        except Exception:
            continue

    if not section_rows:
        return None

    frame = pd.DataFrame(section_rows)
    feature_map = {}
    for column in ["size_of_data", "virtual_address", "entropy", "virtual_size"]:
        series = frame[column]
        feature_map[f"{column}_mean"] = float(series.mean())
        feature_map[f"{column}_max"] = float(series.max())
        feature_map[f"{column}_min"] = float(series.min())
        feature_map[f"{column}_std"] = float(series.std(ddof=0)) if len(series) > 1 else 0.0

    return feature_map


def build_fraud_feature_frame(transaction: dict, ohe, scaler) -> pd.DataFrame:
    tx_type = str(transaction.get("type", "TRANSFER")).upper()
    amount = max(float(transaction.get("amount", 0.0)), 0.0)
    step = int(transaction.get("step", 1) or 1)
    hour = (step - 1) % 24

    base = {
        "amount": amount,
        "hour": float(hour),
        "is_night": 1.0 if (hour >= 22 or hour <= 4) else 0.0,
        "log_amount": float(np.log1p(amount)),
        "is_high_amount": 1.0 if amount > 200000 else 0.0,
        "orig_tx_count_last_24h": float(transaction.get("orig_tx_count_last_24h", 1.0)),
        "orig_amount_sum_last_24h": float(transaction.get("orig_amount_sum_last_24h", amount)),
        "orig_velocity_last_24h": float(
            transaction.get("orig_velocity_last_24h", transaction.get("orig_amount_sum_last_24h", amount))
        ),
        "orig_out_degree": float(transaction.get("orig_out_degree", 1.0)),
        "dest_in_degree": float(transaction.get("dest_in_degree", 1.0)),
        "is_merchant": 1.0 if str(transaction.get("nameDest", "")).startswith("M") else 0.0,
    }

    type_frame = pd.DataFrame([{"type": tx_type}])
    encoded = ohe.transform(type_frame)
    encoded_cols = list(ohe.get_feature_names_out(["type"]))
    encoded_dict = {column: float(encoded[0][index]) for index, column in enumerate(encoded_cols)}

    merged = {**base, **encoded_dict}
    frame = pd.DataFrame([merged])

    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    if scaler_columns:
        for column in scaler_columns:
            if column not in frame.columns:
                frame[column] = 0.0
        frame.loc[:, scaler_columns] = scaler.transform(frame[scaler_columns])

    return frame
