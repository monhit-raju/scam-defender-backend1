import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import TransformerMixin

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)               # remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text.lower())       # alphanumeric only
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(tokens)

def clean_url(url: str) -> str:
    # strip protocol and trailing slash
    u = re.sub(r'^https?://', '', url).rstrip('/')
    return re.sub(r'[^A-Za-z0-9\.]', ' ', u).lower()

class CleanTextTransformer(TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [clean_text(str(x)) for x in X]


