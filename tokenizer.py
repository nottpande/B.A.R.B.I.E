from src.BPE_tokenizer import BPE
import pickle
from tqdm import tqdm

def load_corpus(filepath):
    """Loads the corpus from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    return [sentence.strip() for sentence in corpus]

corpus_file = "Data/corpus.txt"

# Add a progress bar while loading the corpus
print("Loading corpus...")
with tqdm(total=1, desc="Loading Corpus") as pbar:
    corpus = load_corpus(corpus_file)
    pbar.update(1)
print("Corpus Loaded Successfully!")

# Initialize and train BPE tokenizer with progress display
bpe = BPE(corpus, vocab_size=100000)
print("Training BPE tokenizer...")
merges = bpe.train()

# Save the trained BPE model (merges)
print("Saving the Model...")
with tqdm(total=1, desc="Saving Model") as pbar:
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(merges, f)
    pbar.update(1)
print("BPE tokenizer model saved successfully!")
