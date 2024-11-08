from Preprocessing import (
    LowercaseHandler, WhitespaceNormalizer, PunctuationRemover,
    EnglishCharacterRemover, KannadaStemmer, KannadaLemmatizer, EnglishStemmer,
    EnglishLemmatizer, EmptyLineRemover, Padding
)
from BPE_tokenizer import BPE

class TextPreprocessingPipeline:
    def __init__(self, contractions_file, corpus, vocab_size):
        # Initialize all the processing components
        #self.contraction_handler = ContractionHandler(contractions_file)
        self.lowercase_handler = LowercaseHandler()
        self.whitespace_normalizer = WhitespaceNormalizer()
        self.punctuation_remover = PunctuationRemover()
        self.english_character_remover = EnglishCharacterRemover()
        self.kannada_stemmer = KannadaStemmer()
        self.kannada_lemmatizer = KannadaLemmatizer()
        self.english_stemmer = EnglishStemmer()
        self.english_lemmatizer = EnglishLemmatizer()
        self.empty_line_remover = EmptyLineRemover()
        self.bpe_tokenizer = BPE(corpus, vocab_size)
        self.padding = Padding()
        
        # Train the BPE tokenizer
        self.bpe_tokenizer.train()

    def process_text(self, text, language="english", stem=False, lemmatize=False, use_bpe=False):
        """
        Process the text through the pipeline based on specified language and processing options.
        Parameters:
        - text (str): The input text to be processed.
        - language (str): 'english' or 'kannada'.
        - stem (bool): Apply stemming if True.
        - lemmatize (bool): Apply lemmatization if True.
        - use_bpe (bool): Apply BPE tokenization if True.
        
        Returns:
        - str or list: The processed text or BPE tokens.
        """
        # Step 1: Expand contractions (only for English text)
        #if language == "english":
            #text = self.contraction_handler.expand_contractions(text)

        # Step 2: Convert to lowercase
        text = self.lowercase_handler.to_lowercase(text)
        
        # Step 3: Normalize whitespaces
        text = self.whitespace_normalizer.normalize_whitespaces(text)

        # Step 4: Remove punctuation
        text = self.punctuation_remover.remove_punctuation(text)

        # Step 5: Remove English characters if processing Kannada text
        if language == "kannada":
            text = self.english_character_remover.remove_characters(text)

        # Step 6: Apply stemming or lemmatization based on language and user choice
        if language == "english":
            if stem:
                text = self.english_stemmer.stem_sentence(text)
            elif lemmatize:
                text = self.english_lemmatizer.lemmatize_sentence(text)
        elif language == "kannada":
            if stem:
                text = self.kannada_stemmer.stem_sentence(text)
            elif lemmatize:
                text = self.kannada_lemmatizer.lemmatize_sentence(text)

        # Step 7: Remove empty lines (if text is empty after all processing)
        if self.empty_line_remover.is_empty_line(text):
            return None

        # Step 8: Apply BPE tokenization
        if use_bpe:
            tokens = self.bpe_tokenizer.tokenize(text)
            return tokens  # Return BPE tokens instead of processed text
        
        return text  # Return processed text if BPE is not used

    def pad_texts(self, tokenized_texts, pad_token="<PAD>"):
        """
        Pad a list of tokenized sentences to the length of the longest sentence.
        
        Parameters:
        - tokenized_texts (list of list of str): List of tokenized sentences.
        - pad_token (str): The token to use for padding.
        
        Returns:
        - list of list of str: The list of padded sentences.
        """
        padded_sentences = self.padding.pad_sentences(tokenized_texts)
        
        # Print padded sentences in a readable format
        for i, padded_text in enumerate(padded_sentences):
            print(f"Sentence {i+1}: {' '.join(padded_text)}")
        
        return padded_sentences


# Example usage:
if __name__ == "__main__":
    contractions_file = 'path_to_contractions.json'
    corpus = ["I've been running to catch up!", "Hello, this is a test corpus."]
    vocab_size = 100

    pipeline = TextPreprocessingPipeline(contractions_file, corpus, vocab_size)

    # Process English and Kannada sentences separately
    english_texts = ["I've been running to catch up!", "Another test sentence."]
    kannada_texts = ["ಕನ್ನಡ ವಾಕ್ಯ", "ಇದು ಇನ್ನೊಂದು ವಾಕ್ಯ"]

    # Tokenize and process
    tokenized_english = [pipeline.process_text(text, language="english", use_bpe=True) for text in english_texts]
    tokenized_kannada = [pipeline.process_text(text, language="kannada", use_bpe=True) for text in kannada_texts]

    # Pad the tokenized sentences
    padded_english = pipeline.pad_texts(tokenized_english)
    padded_kannada = pipeline.pad_texts(tokenized_kannada)

    print("Padded English Sentences:", padded_english)
    print("Padded Kannada Sentences:", padded_kannada)
