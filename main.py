import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

# Specify the path to your custom stopwords file
custom_stopwords_file = 'stopwords.txt'

# Check and download punkt and stopwords resources

# Load custom stopwords from the file
with open(custom_stopwords_file, 'r') as file:
    custom_stopwords = set(word.strip() for word in file)

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_document(document):
    # Tokenization
    tokens = word_tokenize(document)

    # Stopword removal
    stop_words = set(custom_stopwords)
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Filtering: Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]

    # Stemming (Optional) document & documentation -> document
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

# Example usage
document_text = "This is a sample document with some custom stopwords and punctuation and zzcharles!"
processed_tokens = preprocess_document(document_text)
print(processed_tokens)