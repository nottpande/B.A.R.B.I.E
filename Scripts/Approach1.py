from src.Preprocessing import TextNormalizerEnglish, TextNormalizerKannada
from src.RandomIndexing import RandomIndexing, BilingualCorpus, BilingualVectorAligner, Translator
import pandas as pd
import regex as re

def tokenize(sentence):
    # Tokenize the sentence by capturing letters (including Kannada characters) and punctuation
    return re.findall(r'\p{L}+|[^\w\s]', sentence, flags=re.UNICODE)

if __name__ == "__main__":
    # Step 1: Read the CSV File
#     english_sentences = [
#     "Hello, how are you?",
#     "I am fine, thank you.",
#     "What is your name?",
#     "My name is John.",
#     "Where do you live?",
#     "I live in Bangalore.",
#     "I like reading books.",
#     "The weather is nice today.",
#     "Do you speak Kannada?",
#     "Yes, I can speak Kannada."
# ]

#     kannada_sentences = [
#     "ಹೇಲೋ, ನೀವು ಹೇಗಿದ್ದೀರಾ?",
#     "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ, ಧನ್ಯವಾದಗಳು.",
#     "ನಿನ್ನ ಹೆಸರು ಏನು?",
#     "ನನ್ನ ಹೆಸರು ಜಾನ್.",
#     "ನೀವು ಎಲ್ಲಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
#     "ನಾನು ಬೆಂಗಳೂರು ನಲ್ಲಿ ವಾಸಿಸುತ್ತೇನೆ.",
#     "ನಾನು ಪುಸ್ತಕಗಳನ್ನು ಓದಲು ಇಷ್ಟಪಡುತ್ತೇನೆ.",
#     "ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ.",
#     "ನೀವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೀರಾ?",
#     "ಹೌದು, ನಾನು ಕನ್ನಡ ಮಾತನಾಡಬಲ್ಲೆ."
# ]

    df = pd.read_csv('./Data/dataset_kag.csv')
    print(f'size of the dataframe that is loaded {df.shape}')
    english_sentences = df['English'].tolist()  # Extract the English sentences column as a list
    kannada_sentences = df['Kannada'].tolist()  # Extract the Kannada sentences column as a list

    # with open('./Data/dataset_kag.csv', 'r') as file:
    #     csv_reader = csv.reader(file)
    #     next(csv_reader)  # Skip header if present
    #     for row in csv_reader:
    #         english_sentences.append(row[0].split())  # Tokenize English sentences
    #         kannada_sentences.append(row[1].split())  # Tokenize Kannada sentences

    # Step 2: Preprocess English and Kannada Sentences
    english_normalizer = TextNormalizerEnglish('./Data/english_contractions.json')
    kannada_normalizer = TextNormalizerKannada()

    # Step 3: Build Vocabularies
    corpus = BilingualCorpus(english_sentences, kannada_sentences)
    eng_vocab, kan_vocab = corpus.build_vocab()
    print(eng_vocab)

    # Step 4: Random Indexing for Context Vectors
    english_indexing = RandomIndexing(eng_vocab)
    kannada_indexing = RandomIndexing(kan_vocab)
    seed_pairs = [
                    ("Scientist", "ಸಂಶೋಧಕ"),
                    ("Truth", "ಸತ್ಯ"),
                    ("Gujarat", "ಸೂರತ್"),
                    ("Vikas", "ವಿಕಾಸ"),
                    ("Loot", "ಕಳ್ಳತನ"),
                    ("Read", "ಓದು"),
                    ("Dead", "ಮೃತ"),
                    ("Phone", "ಫೋನ್"),
                    ("Battery", "ಬ್ಯಾಟರಿ"),
                    ("Satan", "ಸೈತಾನ"),
                    ("Prime", "ಪ್ರಧಾನಿ"),  # Simplified from "Prime Minister"
                    ("Corruption", "ಭ್ರಷ್ಟಾಚಾರ"),
                    ("Difference", "ವ್ಯತ್ಯಾಸ"),
                    ("Film", "ಚಿತ್ರ"),
                    ("Shooting", "ಚಿತ್ರೀಕರಣ"),
                    ("King", "ಅರಸ"),
                    ("Foreigner", "ಅನ್ಯ ದೇಶದವನು"),
                    ("Exile", "ಸೆರೆಹಿಡಿಯಲ್ಪಟ್ಟವನು"),
                    ("UN", "ವಿಶ್ವ"),  # Simplified from "UN General Assembly"
                    ("HD", "ಹೈ"),  # Simplified from "HD Recording"
                    ("Discussion", "ಚರ್ಚೆ"),
                    ("Director", "ನಿರ್ದೇಶಕರು"),  # Simplified
                    ("Buses", "ಬಸ್"),
                    ("RSS", "ಆರೆಸ್ಸೆಸ್"),
                    ("BJP", "ಬಿಜೆಪಿ"),
                    ("Tradition", "ಪರಂಪರೆ"),
                    ("Youth", "ಯುವಕರು"),  # From "Kashmiri youth"
                    ("Army", "ಸೇನೆ"),  # From "Indian Army"
                    ("Salary", "ವೇತನ"),
                    ("Village", "ಹಳ್ಳಿ"),
                    ("Cricket", "ಕ್ರಿಕೆಟ್"),  # Simplified from "Cricket captain"
                    ("Monsoon", "ಮಳೆ"),
                    ("Rubbish", "ಕೊಳಚೆ"),
                    ("Traffic", "ಸೊಳ್ಳೆಗಳು"),  # Adjusted to match context
                    ("Economy", "ಆರ್ಥಿಕತೆ"),
                    ("Carrots", "ಪಾರ್ಸ್ಲಿ"),  # Potential typo - please verify!
                    ("Potatoes", "ತುಳಸಿ"),  # Potential typo - please verify!
                    ("Fitness", "ಫಿಟ್‌ನೆಸ್"),  # Simplified from "Fitness Band"
                    ("Police", "ಪೋಲಿಸರು"),
                    ("Hospital", "ಆಸ್ಪತ್ರೆ"),
                    ("Writing", "ಬರೆಯುವುದು"),
                    ("Channels", "ಚಾನೆಲ್‌ಗಳು"),
                    ("Songs", "ಹಾಡುಗಳು"),
                    ("Resolution", "ರೆಸೊಲ್ಯೂಶನ್"),
                    ("Living", "ಜೀವನಮಟ್ಟ"),
                    ("Plan", "ಯೋಜನೆ"),
                    ("Respected", "ಗೌರವ"),
                    ("Picture", "ಫೋಟೋ"),
                    ("Beautiful", "ಸುಂದರ"),
                    ("Consistent", "ಸಂಕೇತ")
    ]


    
    # Update context vectors for English sentences
    for sentence in english_sentences:
        # Tokenize sentence using the updated regex
        tokens = tokenize(sentence)
        print(f"Updating context vectors for tokens: {tokens}")
        english_indexing.update_context_vectors(tokens)

    # Update context vectors for Kannada sentences
    for sentence in kannada_sentences:
        tokens = sentence.split()
        print(f"Updating context vectors for tokens: {tokens}")
        kannada_indexing.update_context_vectors(tokens)

    # Step 5: Learn Transformation Matrix
    aligner = BilingualVectorAligner(english_indexing.context_vectors, kannada_indexing.context_vectors, seed_pairs)
    print("Aligning Vectors, training aligners")
    aligner.learn_transformation_matrix()

    # Step 6: Create Translator and save the model.
    translator = Translator(aligner, english_indexing, kannada_indexing)
    print("Saving the Model")
    translator.save_model('./Models/translator_model_eng2kan.pkl')
    print("Model saved successfully!")
