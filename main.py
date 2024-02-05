import os
import csv
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

nltk.download('punkt')

# Specify the path to your custom stopwords file
custom_stopwords_file = 'stopwords.txt'

# Load custom stopwords from the file
with open(custom_stopwords_file, 'r') as file:
    custom_stopwords = set(word.strip() for word in file)

def preprocess_document(document, docno):
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

    return [(token, docno) for token in set(tokens)]  # Use set to remove duplicate tokens

def process_collection(folder_path):
    # Create a dictionary to store tokens, corresponding docnos, and count of appearances
    tokens_dict = {}

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()

                # Extract text inside <TEXT> brackets
                start_index = file_content.find('<TEXT>') + len('<TEXT>')
                end_index = file_content.find('</TEXT>', start_index)
                text_inside_text_tag = file_content[start_index:end_index]

                # Extract all <DOCNO> values
                docno_tags = file_content.split('<DOCNO>')[1:]
                for docno_tag in docno_tags:
                    docno_end_index = docno_tag.find('</DOCNO>')
                    docno = docno_tag[:docno_end_index].strip()

                    # Preprocess the document and add tokens to the dictionary
                    tokens = preprocess_document(text_inside_text_tag, docno)
                    for token, docno in tokens:
                        if token not in tokens_dict:
                            tokens_dict[token] = {'docnos': set(), 'count': 0}
                        tokens_dict[token]['docnos'].add(docno)
                        tokens_dict[token]['count'] += 1

    return tokens_dict

def export_to_csv(dictionary, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['Token', 'Count', 'DOCNO']
        csv_writer.writerow(header)

        for token, info in dictionary.items():
            count = info['count']
            docnos = ', '.join(info['docnos'])
            csv_writer.writerow([token, count, docnos])

# Example usage
collection_folder = 'coll'  # Change to the actual folder path
output_csv_file = 'output_tokens_dict.csv'  # Change to the desired output file name
output_tokens_dict = process_collection(collection_folder)
export_to_csv(output_tokens_dict, output_csv_file)
