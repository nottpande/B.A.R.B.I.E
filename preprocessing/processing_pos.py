# Load POS tags from a file
def load_pos_tags(file_path):
    pos_tags = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, tag = line.strip().split('\t')  # Each line: "word TAB tag"
            pos_tags[word] = tag
    return pos_tags

# Tokenizer function (same as above, with punctuation handling)
def tokenize_kannada(text):
    punctuations = "।!?.,;:'\"()-"
    for char in punctuations:
        text = text.replace(char, " ")
    tokens = text.split()
    return tokens


# Function to apply POS tagging for a line
def pos_tag_line(line, pos_dict):
    tokens = tokenize_kannada(line)  # Use the same tokenizer
    tagged_tokens = [(word, pos_dict.get(word, "UNK")) for word in tokens]  # Tag words or mark as "UNK"
    return tagged_tokens

# Automate POS tagging for a file
def pos_tag_file(input_file, output_file, pos_file):
    pos_dict = load_pos_tags(pos_file)  # Load POS tags from dictionary
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            tagged_tokens = pos_tag_line(line, pos_dict)
            # Format: "word1_TAG word2_TAG ..." and write to file
            tagged_line = ' '.join([f"{word}_{tag}" for word, tag in tagged_tokens])
            outfile.write(tagged_line + "\n")
    print(f"POS tagging completed. Processed file saved as: {output_file}")

# Specify file paths
input_file_male = "line_index_male.tsv"  # Original Kannada text
input_file_female = "line_index_female.tsv"  # Original Kannada text
output_file_male = "tagged_kannada_text_male.txt"  # Output file with POS tags
output_file_female = "tagged_kannada_text_female.txt"  # Output file with POS tags
pos_file = "kannada_pos_tags.txt"  # POS tag dictionary

# Run the POS tagging process
pos_tag_file(input_file_male, output_file_male, pos_file)
pos_tag_file(input_file_female, output_file_female, pos_file)
