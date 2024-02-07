import os
import json
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

custom_stopwords_file = 'stopwords.txt'
collection_folder = 'coll'
output_json_file = 'tokens.json'

# Load custom stopwords
with open(custom_stopwords_file, 'r') as f:
    custom_stopwords = set(f.read().splitlines())

# Initialize NLTK's PorterStemmer
porter_stemmer = PorterStemmer()

def preprocess_document(document):
    tokens = word_tokenize(document)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in custom_stopwords]
    tokens = [porter_stemmer.stem(token) for token in tokens]
    return tokens

def preprocess_documents(folder):
    preprocessed_docs = {}
    unique_tokens = set()  # Set to keep track of unique tokens
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                document = file.read()
            tokens = preprocess_document(document)
            preprocessed_docs[filename] = tokens
            unique_tokens.update(tokens)  # Update the set with tokens from this document
    return preprocessed_docs, unique_tokens

def index_tokens(preprocessed_docs):
    tokens_dict = {}
    for filename, tokens in preprocessed_docs.items():
        for token in tokens:
            if token not in tokens_dict:
                tokens_dict[token] = {filename: 1}
            else:
                tokens_dict[token][filename] = tokens_dict[token].get(filename, 0) + 1
    return tokens_dict

# Preprocess documents and get unique tokens
preprocessed_docs, unique_tokens = preprocess_documents(collection_folder)

# Index tokens
tokens_dict = index_tokens(preprocessed_docs)

# Add total unique token count at the end
tokens_dict["total_unique_tokens"] = len(unique_tokens)

# Write the tokens dictionary to a JSON file
with open(output_json_file, 'w', encoding='utf-8') as jsonfile:
    json.dump(tokens_dict, jsonfile, indent=4)

print(f"Preprocessing and indexing complete. Token document occurrences and total unique token count written to '{output_json_file}'.")
