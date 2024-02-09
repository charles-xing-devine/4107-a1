import math
import os
import json
import nltk
nltk.download('punkt')
import time

import ssl
import re  # Import the 're' module for regular expressions
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET

start_time = time.time()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

custom_stopwords_file = '4107-a1/stopwords.txt'
collection_folder = '4107-a1/collec'
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
                    # Find all occurrences of <TEXT> tags after the current <DOCNO>
                    text_matches = re.findall(r'<DOCNO>{}</DOCNO>(.*?)<\/TEXT>'.format(re.escape(doc_no)), document_content, re.DOTALL)
                    tokens = []
                    for text in text_matches:
                        # Preprocess each text and update tokens list
                        tokens.extend(preprocess_document(clean_xml_content(text)))
                    tokens = list(tokens)  # Remove duplicates
                    preprocessed_docs[doc_no] = tokens
                    unique_tokens.update(tokens)
    return preprocessed_docs, unique_tokens

def index_tokens(preprocessed_docs):
    tokens_dict = {}
    for doc_no, tokens in preprocessed_docs.items():
        for token in tokens:
            if token not in tokens_dict:
                tokens_dict[token] = {doc_no: 1}
            else:
                if doc_no in tokens_dict[token]:
                    tokens_dict[token][doc_no] += 1
                else:
                    tokens_dict[token][doc_no] = 1
    for token, doc_counts in tokens_dict.items():
        if token == "text":
            for doc_no, count in list(doc_counts.items()):
                if count == 1:
                    del tokens_dict[token][doc_no]  # Remove the token count if it is 1
                else:
                    tokens_dict[token][doc_no] -= 1  # Decrement the count by 1 if more than 1
    tokens_dict.pop("", None)
    return tokens_dict

def idf_calculation(query): 
    arr = [] 
    for token in query: 
        if token in tokens_dict: 
            arr.append(math.log2(number_of_docs/len(tokens_dict[token]))) 
            print(arr)
    return arr

def tf_calculation(query): 
    arr = []
    tmp = []
    res = []
    fin = []
    for token in query: 
        if token in tokens_dict:
            arr = list(tokens_dict[token].keys())
            for i in range(len(arr)):
                if arr[i] not in tmp:
                    tmp.append(arr[i])
                    fin.append(arr[i])
                    for j in query:
                        if j in tokens_dict:
                            fin.append(tokens_dict[j].get(arr[i]))
                    res.append(fin)
                    fin = []

                print(res)
    print(max(res[0][1:]))
    return arr

def tf_idf_score(idf, tf): 
    return tf_calculation(tf) * idf_calculation(idf)

def cosine_calculator(): 
    return 0
    
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

# Record the end time after writing the JSON file
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to load and process JSON file: {elapsed_time} seconds")

tf_calculation(preprocess_document("office vietnamese govern"))