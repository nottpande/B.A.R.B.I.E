import re

# Function to load stopwords from a file
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())  # Add each stopword to the set
    return stopwords

# Function to tokenize Kannada text with punctuation handling
def tokenize_kannada(text):
    punctuations = "।!?.,;:'\"()-"
    for char in punctuations:
        text = text.replace(char, " ")
    tokens = text.split()
    return tokens

# Function to remove English alphabets from a given line of text
def remove_english_characters(line):
    """
    This function removes all English alphabets (both lowercase and uppercase) from the input text.
    It uses a regular expression to substitute any English characters with an empty string.
    """
    return re.sub(r'[A-Za-z]', '', line)

# Function to remove stopwords from a given line of text
def remove_stopwords_line(line, stopword_list):
    tokens = tokenize_kannada(line)
    filtered_tokens = [word for word in tokens if word not in stopword_list]
    return ' '.join(filtered_tokens)

# Function to normalize whitespaces in a given line
def normalize_whitespaces(line):
    """
    This function normalizes multiple consecutive whitespaces into a single space.
    It also removes leading and trailing whitespaces.
    """
    return ' '.join(line.split())

# Function to check if a line is empty after processing
def is_empty_line(line):
    """
    This function checks if the line is empty or contains only whitespaces after processing.
    """
    return len(line.strip()) == 0

# Automate stopword removal, English alphabet removal, and normalization for an entire file
def process_kannada_file(input_file, output_file, stopword_file):
    stopword_list = load_stopwords(stopword_file)  # Load stopwords from the file
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Step 1: Remove English alphabets from the line
            no_english_line = remove_english_characters(line)
            
            # Step 2: Remove stopwords from each line
            filtered_line = remove_stopwords_line(no_english_line, stopword_list)
            
            # Step 3: Normalize whitespaces in the filtered line
            normalized_line = normalize_whitespaces(filtered_line)
            
            # Step 4: Skip empty lines after processing
            if not is_empty_line(normalized_line):
                outfile.write(normalized_line + "\n")  # Write the cleaned line to output file
    
    print(f"Stopword removal, English alphabet removal, and text normalization completed. Processed file saved as: {output_file}")

# Specify file paths
input_file_manual = "kannada_sentences.txt"          # Input file containing 4000+ lines of Kannada text
output_file_manual = "processed_kannada_text_manual.txt"  # Output file for storing cleaned text
stopword_file = "kannada_words.txt"     # Stopword file

# Run the stopword removal, English character removal, and text normalization process
process_kannada_file(input_file_manual, output_file_manual, stopword_file)
