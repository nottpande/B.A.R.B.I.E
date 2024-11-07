'''
# Specify file paths for input and output
from src.Preprocessing import TestPPEnglish
from src.Preprocessing import TestPPKannada

input_eng = "PreProcess_Testing/manual.txt"              # Input file containing English sentences
input_kan = "PreProcess_Testing/kan.txt"
output_file = "PreProcess_Testing/manual_o.txt"           # Output file for storing the cleaned and preprocessed text
stopword_eng = "Data/english_stopwords.txt"  # Optional stopword file (if available)
contraction_file = "Data/english_contractions.json"   # Optional contraction file
stopword_kan  = "Data/kannada_words.txt"

# Create an instance of the TestPP class and run the processing function
text_processor_eng = TestPPEnglish(input_eng, output_file, stopword_eng, contraction_file)
text_processor_eng.process_english_file()

text_processor_kan = TestPPKannada(input_kan, output_file, stopword_kan)
text_processor_kan.process_kannada_file()

'''

'''
from src.Preprocessing import EnglishLemmatizer
lem = EnglishLemmatizer()
output = lem.lemmatize_sentence("He is going to school")
print(output)
'''

'''
from src.Transformers import Embedding,PositionalEncoding,Encoder
import torch
sentence = "Clap Your Hands and Stomp your feet!"
tokens = sentence.lower().split()
vocab = {word: idx for idx, word in enumerate(set(tokens))}  # Simple vocabulary
token_ids = torch.tensor([[vocab[word] for word in tokens]], dtype=torch.long)  # Shape: (1, len(tokens))
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

vocab_size = len(vocab)
embedding_dim = 256
max_seq_len = 50
num_layers = 2
expansion_factor = 4
n_heads = 2

embedding_layer = Embedding(vocab_size, embedding_dim)
positional_encoder = PositionalEncoding(max_seq_len, embedding_dim)
encoder = Encoder(num_layers, embedding_dim, expansion_factor, n_heads)

embeddings = embedding_layer(token_ids)
print(embeddings)
pos_enc = positional_encoder(embeddings)
encoding = embeddings + pos_enc
contextual_embeddings = encoder(encoding)
print("Contextual Embedding created!")
print(contextual_embeddings)
print(f'Contextual Embedding shape: {contextual_embeddings.shape}')
contextual_embeddings
'''