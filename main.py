import os
import json
import nltk
import ssl
import re  # Import the 're' module for regular expressions
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

custom_stopwords_file = 'stopwords.txt'
collection_folder = 'collec'
output_json_file = 'tokens.json'

# Load custom stopwords
with open(custom_stopwords_file, 'r') as f:
    custom_stopwords = set(f.read().splitlines())

# Initialize NLTK's PorterStemmer
porter_stemmer = PorterStemmer()

number_of_docs = 79923

##################################

# Define a function to clean XML content using regex
def clean_xml_content(xml_content):
    # Define a regular expression pattern to match lines not encapsulated by <DOCNO> and <TEXT> tags
    pattern = r'<(?!DOCNO|TEXT)[^>]*>.*?</[^>]*>\n?'

    cleaned_xml_content = re.sub(pattern, '', xml_content)
    return cleaned_xml_content

def preprocess_document(document):
    # Tokenize the document
    tokens = word_tokenize(document)
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in custom_stopwords]
    tokens = [porter_stemmer.stem(token) for token in tokens]
    
    return tokens

def preprocess_query(query):
    tokens = word_tokenize(query)
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
                document_content = file.read()
                # Extract <DOCNO> tags using regex
                doc_no_matches = re.findall(r'<DOCNO>(.*?)<\/DOCNO>', document_content, re.DOTALL)
                for doc_no in doc_no_matches:
                    # Find the text between <TEXT> tags that are after the current <DOCNO>
                    text_matches = re.findall(r'<DOCNO>{}</DOCNO>\s*<TEXT>(.*?)<\/TEXT>'.format(doc_no), document_content, re.DOTALL)
                    if text_matches:
                        # Only consider the first match as there should only be one <TEXT> tag per <DOCNO>
                        text = text_matches[0]
                        tokens = preprocess_document(clean_xml_content(text))
                        tokens = list(set(tokens))
                        preprocessed_docs[doc_no] = tokens
                        unique_tokens.update(tokens)  
    return preprocessed_docs, unique_tokens

def index_tokens(preprocessed_docs):
    tokens_dict = {}
    for doc_no, tokens in preprocessed_docs.items():
        print(preprocessed_docs.items())
        for token in tokens:
            if token not in tokens_dict:
                tokens_dict[token] = {doc_no: 1}
            else:
                tokens_dict[token][doc_no] = tokens_dict[token].get(doc_no, 0) + 1
    return tokens_dict

def idf_calculation(query): 

    arr = [] 

    tmp = 0; 

    for token in query: 
        if token in tokens_dict: 
            value = tokens_dict[token]
            arr.append(value) 
            len(value)
            for i in range(len(value)):
                tmp += len(value)

                ## {"AP8080"; 1}

    return arr

def tf_calculation(): 
    return 0

def tf_idf_score(idf, tf): 
    return tf_calculation(tf) * idf_calculation(idf)
    
##################################

# Call preprocess_documents to preprocess the documents
preprocessed_docs, unique_tokens = preprocess_documents(collection_folder)

# Call index_tokens to index the tokens
tokens_dict = index_tokens(preprocessed_docs)

# Add total unique token count at the end
tokens_dict["total_unique_tokens"] = len(unique_tokens)

# Write the tokens dictionary to a JSON file
with open(output_json_file, 'w', encoding='utf-8') as jsonfile:
    json.dump(tokens_dict, jsonfile, indent=4)