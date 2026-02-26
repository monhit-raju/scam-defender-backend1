import re
import numpy as np
import ipaddress
from urllib.parse import urlparse
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import csr_matrix
from collections import Counter

# ── 1) Base URL Lexical Features ────────────────────────────────────────────────
class URLLexicalFeatures(TransformerMixin, BaseEstimator):
    """Simple lexical features from URL: length, dot count, slash count, token count."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for url in X:
            length = len(url)
            dots   = url.count(".")
            slashes= url.count("/")
            tokens = len(re.split(r"\W+", url))
            feats.append([length, dots, slashes, tokens])
        return np.array(feats)


# ── 2) Enhanced URL Features ─────────────────────────────────────────────────────
class EnhancedURLLexicalFeatures(TransformerMixin, BaseEstimator):
    """Extend URLLexicalFeatures with IP‑check, subdomain count, path‑length, '@' and dash counts."""
    def __init__(self, base_lex=None):
        self.base_lex = base_lex or URLLexicalFeatures()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # original lexical features
        base_feats = self.base_lex.transform(X)
        extra = []
        for url in X:
            parsed = urlparse(url)  # split into components :contentReference[oaicite:0]{index=0}
            # is netloc an IP address?
            is_ip = 0
            try:
                host = parsed.netloc.split(":")[0]
                ipaddress.ip_address(host)
                is_ip = 1
            except Exception:
                is_ip = 0
            parts = parsed.netloc.split(".")
            sub_count = max(0, len(parts) - 2)
            path_len  = len(parsed.path)
            at_sign   = url.count("@")
            dash_cnt  = url.count("-")
            extra.append([is_ip, sub_count, path_len, at_sign, dash_cnt])
        # horizontally stack base_feats and extra arrays :contentReference[oaicite:1]{index=1}
        return np.hstack([base_feats, np.array(extra)])


# ── 3) Scam Keyword Counter ──────────────────────────────────────────────────────
class ScamKeywordCounter(TransformerMixin, BaseEstimator):
    """Count occurrences of common scam keywords in text."""
    DEFAULT_KEYWORDS = [
        "free","winner","urgent","prize","click","credit","offer",
        "congratulations","bank","account","password","risk","loan"
    ]
    def __init__(self, keywords=None):
        self.keywords = keywords or self.DEFAULT_KEYWORDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is iterable of strings
        counts = []
        for text in X:
            lc = text.lower()
            counts.append([lc.count(k) for k in self.keywords])
        return np.array(counts)


# ── 4) File Feature Extractor ────────────────────────────────────────────────────
class FileFeatureExtractor(TransformerMixin, BaseEstimator):
    """
    Transform a DataFrame of file‐level numeric features into numpy array.
    E.g. columns: type, malice, generic, trojan, ransomware, worm, backdoor, spyware, rootkit, encrypter, downloader
    """
    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df):
        # preserve column order
        cols = ["type","malice","generic","trojan","ransomware","worm",
                "backdoor","spyware","rootkit","encrypter","downloader"]
        arr = X_df[cols].to_numpy()
        # convert to sparse matrix if needed
        return csr_matrix(arr)




