import random
from collections import defaultdict
import numpy as np
import pickle
import re


class RandomIndexing:
    def __init__(self, vocab, dim=256, sparsity=0.8):
        self.dim = dim
        self.vocab = vocab  # Store the actual vocabulary
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}  # Create word to index mapping
        self.index_vectors = {}
        self.context_vectors = defaultdict(self.create_zero_vector)  # Changed this to call a regular method
        print(f"Initializing RandomIndexing with vocab_size={len(vocab)}, dim={dim}, sparsity={sparsity}")
        self.initialize_index_vectors(sparsity)

    def create_zero_vector(self):
        """Method to create a zero vector of the specified dimension"""
        return np.zeros(self.dim)

    def initialize_index_vectors(self, sparsity):
        print(f"Initializing index vectors for {len(self.vocab)} words...")
        for word in self.vocab:
            vector = np.zeros(self.dim)
            non_zero_indices = random.sample(range(self.dim), int(self.dim * (1 - sparsity)))
            for index in non_zero_indices:
                vector[index] = random.choice([-1, 1])
            self.index_vectors[word] = vector
        print("Index vectors initialized.")

    def update_context_vectors(self, tokens, window_size=3):
        print(f"Updating context vectors for sentence with {len(tokens)} tokens...")
        for i, token in enumerate(tokens):
            if token not in self.vocab:
                print(f"Warning: '{token}' not in vocabulary, skipping...")
                continue
                
            left_context = tokens[max(i - window_size, 0):i]
            right_context = tokens[i + 1:i + 1 + window_size]
            context = left_context + right_context
            
            for neighbor in context:
                if neighbor in self.index_vectors:
                    self.context_vectors[token] += self.index_vectors[neighbor]
                else:
                    print(f"Warning: '{neighbor}' not found in index vectors, skipping...")
        print("Context vectors updated.")


class BilingualCorpus:
    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences
        self.english_vocab = set()
        self.kannada_vocab = set()
        print("Initializing BilingualCorpus...")

    def build_vocab(self):
        print("Building vocabularies for English and Kannada...")
        
        # Initialize vocabularies as sets to ensure unique words
        self.english_vocab = set()
        for sentence in self.english_sentences:
            # Use regex to split words and punctuation
            words = re.findall(r'\w+|[^\w\s]', sentence)  # Match words and punctuation separately
            for word in words:
                self.english_vocab.add(word)

        self.kannada_vocab = set()
        for sentence in self.kannada_sentences:
            for word in sentence.split():
                self.kannada_vocab.add(word)

        print(f"English vocabulary size: {len(self.english_vocab)}")
        print(f"Kannada vocabulary size: {len(self.kannada_vocab)}")
        
        # Return vocabularies as lists
        return list(self.english_vocab), list(self.kannada_vocab)


class BilingualVectorAligner:
    def __init__(self, english_context_vectors, kannada_context_vectors, seed_pairs):
        self.english_vectors = english_context_vectors
        self.kannada_vectors = kannada_context_vectors
        self.seed_pairs = seed_pairs
        self.transformation_matrix = None
        print(f"Initializing BilingualVectorAligner with {len(seed_pairs)} seed pairs.")

    def learn_transformation_matrix(self):
        print("Learning transformation matrix...")
        valid_pairs = [(eng, kan) for eng, kan in self.seed_pairs 
                      if eng in self.english_vectors and kan in self.kannada_vectors]
        print("Valid pairs found:", valid_pairs)
        if not valid_pairs:
            raise ValueError("No valid seed pairs found in the context vectors!")
            
        X = np.array([self.english_vectors[eng] for eng, _ in valid_pairs])
        Y = np.array([self.kannada_vectors[kan] for _, kan in valid_pairs])
        
        if X.shape[0] < X.shape[1]:
            print("Warning: Not enough seed pairs for reliable transformation!")
            
        self.transformation_matrix = np.linalg.lstsq(X, Y, rcond=None)[0]
        print("Transformation matrix learned.")
        return self.transformation_matrix


class Translator:
    def __init__(self, aligner, english_random_indexing, kannada_random_indexing):
        """Initialize the translator with necessary components"""
        self.transformation_matrix = aligner.transformation_matrix  # Store the matrix explicitly
        self.english_random_indexing = english_random_indexing
        self.kannada_random_indexing = kannada_random_indexing
        print("Translator initialized.")

    def save_model(self, file_path):
        """Save the model components to a file"""
        print(f"Saving translator model to: {file_path}")
        model_data = {
            'transformation_matrix': self.transformation_matrix,
            'english_indexing': self.english_random_indexing,
            'kannada_indexing': self.kannada_random_indexing
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

    @staticmethod
    def load_model(file_path):
        """Load the model components from a file"""
        print(f"Loading translator model from: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a minimal aligner with just the transformation matrix
        aligner = BilingualVectorAligner(None, None, [])
        aligner.transformation_matrix = data['transformation_matrix']
        
        return Translator(
            aligner,
            data['english_indexing'],
            data['kannada_indexing']
        )

    def translate(self, english_sentence):
        """Translate an English sentence to Kannada"""
        print(f"Translating sentence: {' '.join(english_sentence)}")
        translated_sentence = []
        
        for word in english_sentence:
            if word not in self.english_random_indexing.context_vectors:
                print(f"Warning: '{word}' not found in English context vectors")
                translated_sentence.append("<unknown>")
                continue
                
            eng_vector = self.english_random_indexing.context_vectors[word]
            kan_vector = self.aligner.transform_vector(eng_vector)
            translated_word = self.find_closest_match(kan_vector)
            translated_sentence.append(translated_word)
            
        print("Translation complete.")
        return ' '.join(translated_sentence)

    def find_closest_match(self, vector):
        """Find the closest matching Kannada word for a given vector"""
        best_match = None
        best_similarity = -float('inf')
        
        for kannada_word in self.kannada_random_indexing.vocab:
            kan_vector = self.kannada_random_indexing.context_vectors[kannada_word]
            norm_vector = np.linalg.norm(vector)
            norm_kan_vector = np.linalg.norm(kan_vector)
            
            if norm_vector == 0 or norm_kan_vector == 0:
                continue
                
            similarity = np.dot(vector, kan_vector) / (norm_vector * norm_kan_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = kannada_word
                
        return best_match or "<unknown>"