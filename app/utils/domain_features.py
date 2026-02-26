import ssl, socket, tldextract
import pandas as pd
from datetime import datetime
from sklearn.base import TransformerMixin

# extract domain
def extract_domain(url):
    ext = tldextract.extract(url)
    return ext.domain + ('.'+ext.suffix if ext.suffix else '')

# simple Transformer: no WHOIS/VT for speed
class DomainFeatureExtractor(TransformerMixin):
    def __init__(self, whitelist=None, vt_api_key=None):
        self.whitelist = whitelist or []
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        df = pd.DataFrame({'url':X})
        df['domain']=df.url.map(extract_domain)
        df['is_whitelist']=df.domain.isin(self.whitelist).astype(int)
        return df[['is_whitelist']].to_numpy()








