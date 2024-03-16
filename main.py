import math
import os
import nltk
nltk.download('punkt')
import time
import numpy as np
import ssl
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from numpy.linalg import norm
import json

from sentence_transformers import SentenceTransformer, util
import torch

start_time = time.time()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

custom_stopwords_file = '4107-a1/stopwords.txt'
collection_folder = '4107-a1/collection'
output_tokens_json_file = 'tokens.json'
file_path = '4107-a1/test_queries.txt'

# Load custom stopwords
with open(custom_stopwords_file, 'r') as f:
    custom_stopwords = set(f.read().splitlines())

# Initialize NLTK's PorterStemmer
porter_stemmer = PorterStemmer()

number_of_docs = 79923

text_dict = {}

##################################
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in custom_stopwords]
    tokens = [porter_stemmer.stem(token) for token in tokens]
    return tokens

def index_topic_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    topics = re.findall(r'<top>(.*?)</top>', content, re.DOTALL)
    indexed_tokens = {}

    for topic in topics:
        num = re.search(r'<num>\s*(\d+)\s*', topic)
        title = re.search(r'<title>\s*(.*?)\s*\n', topic)
        desc = re.search(r'<desc>\s*.*?\s*(.*?)\s*\n', topic, re.DOTALL)
        if num and title and desc:
            num = num.group(1).strip()
            title_text = title.group(1).strip()
            desc_text = desc.group(1).strip()
            # Preprocess title and description
            title_tokens = preprocess_text(title_text)
            desc_tokens = preprocess_text(desc_text)

            # Token frequency indexing
            indexed_tokens[num] = {
                'title': {token: title_tokens.count(token) for token in set(title_tokens)},
                'desc': {token: desc_tokens.count(token) for token in set(desc_tokens)}
            }        
    return indexed_tokens

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

def extract_text(content):
    # Join the list into a single string if it's provided as a list
    if isinstance(content, list):
        content = ' '.join(content)
    
    # Regular expression to find content following the <TEXT> tag
    pattern = r'<TEXT>(.*)'
    
    # Search for the pattern and extract the content
    match = re.search(pattern, content, re.DOTALL)
    
    # If a match is found, process the content
    if match:
        extracted_content = match.group(1).strip()  # .strip() removes leading and trailing whitespace
        
        # Remove \n and \t from the extracted content
        cleaned_content = re.sub(r'[\n\t]+', ' ', extracted_content)
        
        return cleaned_content
    else:
        return ""

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
                    text_dict[doc_no] = extract_text(text_matches)
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

def clean_query(query):
    seen = set()
    unique_query = [x for x in query if not (x in seen or seen.add(x))] #returns a list of all words in a query without duplicates in the same order
    query_counts = [query.count(word) for word in unique_query] #returns a list with the count of all words in the query in the same order 
    return unique_query, query_counts

def neural_retrieval1(): 
    return 0; 

##################################
# Index the tokens
indexed_tokens = index_topic_tokens(file_path)

# Call preprocess_documents to preprocess the documents
preprocessed_docs, unique_tokens = preprocess_documents(collection_folder)

# Call index_tokens to index the tokens
tokens_dict = index_tokens(preprocessed_docs)

# Add total unique token count at the end
tokens_dict["total_unique_tokens"] = len(unique_tokens)

# Record the end time after writing the JSON file
end_time = time.time()

documents = list(text_dict.values())  # Convert dict_values to a list
query = "Vietnam arsonist"
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
most_similar_docs_indices = cosine_scores.argsort(descending=True)

for index in most_similar_docs_indices[0][:5]:  # Adjust the slice for the number of top documents you want
    print(f"Document index: {index.item()}, Similarity Score: {cosine_scores[0, index].item()}")

#Calculate the result over the query title
#get_query_title("run_4")

#Calculate the result over the query title and description
#get_query_title_desc("Run_title_description")

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Time taken to load and process JSON file: {elapsed_time} seconds")

