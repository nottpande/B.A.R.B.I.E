import re
from collections import defaultdict
from tqdm import tqdm

class BPE:
    """Byte-Pair Encoding: Subword-based tokenization algorithm."""
    
    def __init__(self, corpus, vocab_size):
        """Initialize BPE tokenizer."""
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}

    def train(self):
        """Train BPE tokenizer."""
        print("Computing word frequencies...")
        # Compute the frequencies of each word in the corpus with progress display
        for text in tqdm(self.corpus, desc="Processing corpus", unit=" sentence"):
            words = text.split()  # Assuming text is already preprocessed
            for word in words:
                self.word_freqs[word] += 1

        # Compute the base vocabulary of all characters in the corpus
        alphabet = list(set(''.join(self.word_freqs.keys())))
        alphabet.sort()

        # Add the special token </w> at the beginning of the vocabulary
        vocab = ["</w>"] + alphabet.copy()

        # Split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        print("Starting BPE merges...")
        # Merge the most frequent pair iteratively until the vocabulary size is reached
        with tqdm(total=self.vocab_size, desc="Merging pairs", unit=" merge") as pbar:
            while len(vocab) < self.vocab_size:
                pair_freqs = self.compute_pair_freqs()

                # Find the most frequent pair
                best_pair = max(pair_freqs, key=pair_freqs.get, default=None)
                if best_pair is None:
                    print("Exhaustion reached: No more merges possible before reaching desired vocabulary size.")
                    break

                # Merge the most frequent pair
                self.splits = self.merge_pair(*best_pair)
                self.merges[best_pair] = best_pair[0] + best_pair[1]
                vocab.append(best_pair[0] + best_pair[1])

                pbar.update(1)  # Update the progress bar after each merge

        if len(vocab) < self.vocab_size:
            print(f"Training stopped early: Vocabulary size of {len(vocab)} reached due to exhaustion.")
        else:
            print(f"Training completed: Vocabulary size of {self.vocab_size} reached.")
            
        return self.merges

    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        """Merge the given pair."""
        merged_token = a + b
        for word in self.word_freqs:
            split = self.splits[word]
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(merged_token)  # Merge the pair
                    i += 2  # Skip the next character as it's merged
                else:
                    new_split.append(split[i])
                    i += 1
            self.splits[word] = new_split
        return self.splits

    def tokenize(self, text):
        """Tokenize a given text with trained BPE tokenizer."""
        print("Tokenizing text...")
        words = text.split()  # Assuming text is already preprocessed
        splits_text = [[c for c in word] for word in words]

        # Wrap tokenization process in tqdm to show progress
        for pair, merge in tqdm(self.merges.items(), desc="Applying merges", unit=" pair"):
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits_text[idx] = split

        result = sum(splits_text, [])
        return result
