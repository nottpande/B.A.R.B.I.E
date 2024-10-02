import string

# Function to load stopwords from a file
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip().lower())  # Add each stopword to the set and convert to lowercase
    return stopwords

# Tokenizer function to split the text into words and handle punctuation
def tokenize_english(text):
    """
    Tokenizes the input text, removing punctuation and splitting it into words.
    """
    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    tokens = text.split()
    return tokens

# Function to remove stopwords from a line of text
def remove_stopwords_line(line, stopword_list):
    """
    Takes a line and removes any stopwords present in the stopword list.
    """
    tokens = tokenize_english(line)
    filtered_tokens = [word for word in tokens if word not in stopword_list]
    return ' '.join(filtered_tokens)

# Function to normalize whitespaces in a given line
def normalize_whitespaces(line):
    """
    This function normalizes multiple consecutive whitespaces into a single space.
    It also removes leading and trailing whitespaces.
    """
    return ' '.join(line.split())

# Function to convert text to lowercase
def to_lowercase(line):
    """
    Converts all characters in a given line to lowercase.
    """
    return line.lower()

# Function to check if a line is empty after processing
def is_empty_line(line):
    """
    Checks if a line is empty or contains only whitespaces after processing.
    """
    return len(line.strip()) == 0

# Automate the cleaning and preprocessing for an entire file
def process_english_file(input_file, output_file, stopword_file=None):
    """
    Processes the English text file by normalizing it, removing punctuation, lowercasing,
    and optionally removing stopwords.
    """
    stopword_list = load_stopwords(stopword_file) if stopword_file else set()  # Load stopwords if a file is provided

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Step 1: Convert text to lowercase
            lowercase_line = to_lowercase(line)
            
            # Step 2: Remove stopwords (if stopword list is provided)
            if stopword_list:
                filtered_line = remove_stopwords_line(lowercase_line, stopword_list)
            else:
                filtered_line = lowercase_line

            # Step 3: Normalize whitespaces
            normalized_line = normalize_whitespaces(filtered_line)
            
            # Step 4: Skip empty lines
            if not is_empty_line(normalized_line):
                outfile.write(normalized_line + "\n")  # Write the cleaned line to the output file
    
    print(f"Text processing completed. Processed file saved as: {output_file}")

# Specify file paths for input and output
input_file = "english_sentences.txt"              # Input file containing English sentences
output_file = "processed_english_text.txt"    # Output file for storing the cleaned and preprocessed text
stopword_file = "english_stopwords.txt"       # Optional stopword file (if available)

# Run the text processing function
process_english_file(input_file, output_file, stopword_file)
