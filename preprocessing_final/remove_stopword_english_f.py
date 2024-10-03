# remove_stopwords_line.py
from punc_english_f import punc_english

def remove_stopwords_line(line, stopword_file):
    """
    Takes a line and removes any stopwords present in the stopword list.
    """
    stopwords_list = set()
    with open(stopword_file, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords_list.add(line.strip().lower())  # Add each stopword to the set and convert to lowercase
    print(f"Loaded {len(stopwords_list)} stopwords from {stopword_file}")
    tokens = punc_english(line)
    filtered_tokens = [word for word in tokens if word not in stopwords_list]
    filtered_line = ' '.join(filtered_tokens)
    print(f"Filtered line: {filtered_line}")
    return filtered_line

# Example usage
if __name__ == "__main__":
    line = "This is a sample sentence for testing stopword removal."
    stopword_file = "english_stopwords.txt"  # Modify this path accordingly
    remove_stopwords_line(line, stopword_file)
