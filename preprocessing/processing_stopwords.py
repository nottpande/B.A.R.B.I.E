# Read stopwords from file
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())  # Add each stopword to the set
    return stopwords

# Tokenizer function (same as above, with punctuation handling)
def tokenize_kannada(text):
    punctuations = "।!?.,;:'\"()-"
    for char in punctuations:
        text = text.replace(char, " ")
    tokens = text.split()
    return tokens

# Function to remove stopwords
def remove_stopwords_line(line, stopword_list):
    tokens = tokenize_kannada(line)
    filtered_tokens = [word for word in tokens if word not in stopword_list]
    return ' '.join(filtered_tokens)

# Automate for an entire file
def process_kannada_file(input_file, output_file, stopword_file):
    stopword_list = load_stopwords(stopword_file)  # Load stopwords from file
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Remove stopwords from each line and write to output
            filtered_line = remove_stopwords_line(line, stopword_list)
            outfile.write(filtered_line + "\n")  # Write the filtered line
    print(f"Stopword removal completed. Processed file saved as: {output_file}")

# Specify file paths
input_file = "line_index_male.tsv"          # Input file containing 4000+ lines of Kannada text
output_file = "processed_kannada_text.txt"  # Output file for storing cleaned text
stopword_file = "kannada_words.txt"     # Stopword file

# Run the stopword removal process
process_kannada_file(input_file, output_file, stopword_file)