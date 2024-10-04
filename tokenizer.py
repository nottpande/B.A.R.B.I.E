from src.BPE_tokenizer import BPE
import pickle
from tqdm import tqdm

def load_corpus(filepath):
    """Loads the corpus from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    return [sentence.strip() for sentence in corpus]

#sample corpus, to check the working of the model
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "ನಾನು ಕನ್ನಡದಲ್ಲಿದ್ದುಕೊಂಡು ವರ್ತಮಾನಗಳನ್ನು ಕಲಿಯುತ್ತೇನೆ.",  # "I am learning tenses in Kannada."
    "Machine translation is a challenging task.",
    "ಭಾಷಾಂತರಣವು ಒಂದು ಸಂಕೀರ್ಣ ಕಾರ್ಯವಾಗಿದೆ.",  # "Translation is a complex task."
    "Artificial Intelligence is transforming industries.",
    "ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆ ಕೈಗಾರಿಕೆಗಳನ್ನು ಪರಿವರ್ತಿಸುತ್ತಿದೆ."  # "Artificial intelligence is transforming industries."
]

'''

# Loading the corpus dataset (to train the model)

corpus_file = "Data/corpus.txt"

# Add a progress bar while loading the corpus
print("Loading corpus...")
with tqdm(total=1, desc="Loading Corpus") as pbar:
    corpus = load_corpus(corpus_file)
    pbar.update(1)
print("Corpus Loaded Successfully!")

'''

# Initialize and train BPE tokenizer with progress display
bpe = BPE(corpus, vocab_size=10000) #10000 for the sample corpus, and #100,000 for the dataset
print("Training BPE tokenizer...")
merges = bpe.train()

print("Performing the sample tokenization")
text_en = "I love machine learning."
text_kn = "ನಾನು ಯಂತ್ರ ಕಲಿಕೆಯ ಬಗ್ಗೆ ಪ್ರೀತಿಸುತ್ತೇನೆ."
tokenized_text_en = bpe.tokenize(text_en)
tokenized_text_kn = bpe.tokenize(text_kn)

print("Tokenized text english: \n",tokenized_text_en)
print("Tokenized text Kannada: \n",tokenized_text_kn)

'''

# Save the trained BPE tokenizer model

print("Saving the Model...")
with tqdm(total=1, desc="Saving Model") as pbar:
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(merges, f)
    pbar.update(1)
print("BPE tokenizer model saved successfully!")

'''