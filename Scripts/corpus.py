import requests
import re

def download_text(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text.splitlines()

def remove_leading_numbers(text):
    return re.sub(r'^\d+\s+', '', text)

eng_url = "https://raw.githubusercontent.com/nottpande/B.A.R.B.I.E/refs/heads/main/preprocessing_final/processed_english_text.txt"
kan_url = "https://raw.githubusercontent.com/nottpande/B.A.R.B.I.E/refs/heads/main/preprocessing_final/processed_kannada_text_manual.txt"

try:
    print("Downloading the Kannada Statements...")
    kannada_sentences = download_text(kan_url)
    print("Cleaning the statements...")
    clean_kan = [remove_leading_numbers(sentence) for sentence in kannada_sentences]
except Exception as e:
    print("Error faced during extraction of Kannada sentence \n")
    print(e)

try:
    print("Downloading the English Statements...")
    english_sentences = download_text(eng_url)
    print("Cleaning the statements...")
    clean_eng = [remove_leading_numbers(sentence) for sentence in english_sentences]
except Exception as e:
    print("Error faced during extraction of English sentence \n")
    print(e)

combined_corpus = [sentence.strip() for sentence in clean_eng + clean_kan]

with open('Data/corpus.txt', 'w', encoding='utf-8') as c_file:
    for sentence in combined_corpus:
        c_file.write(sentence + '\n')
print("Combined corpus saved successfully!")