import string
import re
import json

class ContractionHandler:
    def __init__(self, contractions_file):
        # Load contractions from the specified JSON file
        with open(contractions_file, 'r', encoding='utf-8') as f:
            self.contractions_dict = json.load(f)

    def expand_contractions(self, text):
        """
        Expands contractions in the given text.
        """
        words = text.split()
        expanded_text = []

        for word in words:
            # Remove punctuation for contraction checks
            clean_word = word.strip(string.punctuation).lower()
            if clean_word in self.contractions_dict:
                # Replace contraction with its expanded form
                expanded_word = self.contractions_dict[clean_word]
                # Maintain original word's punctuation
                expanded_text.append(expanded_word + word[len(clean_word):])  # Add any trailing punctuation back
            else:
                expanded_text.append(word)  # Keep the word unchanged if no contraction found

        return ' '.join(expanded_text)


# Class for lowercasing text
class LowercaseHandler:
    """
    This class handles conversion of text to lowercase.
    """
    def to_lowercase(self, text):
        return text.lower()


# Class for removing stopwords
class StopwordRemover:
    """
    This class removes stopwords from the text.
    """
    def __init__(self, stopword_file=None):
        if stopword_file:
            self.stopwords = self.load_stopwords(stopword_file)
        else:
            self.stopwords = set()
    
    def load_stopwords(self, file_path):
        stopwords = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip().lower())  # Add each stopword to the set and convert to lowercase
        return stopwords

    def remove_stopwords(self, text):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in self.stopwords]
        return ' '.join(filtered_tokens)


# Class for normalizing whitespaces
class WhitespaceNormalizer:
    """
    This class handles normalization of whitespaces in text.
    """
    def normalize_whitespaces(self, text):
        return ' '.join(text.split())


# Class for removing punctuation
class PunctuationRemover:
    """
    This class removes punctuation marks from the text.
    """
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))


class EnglishCharacterRemover:
    def __init__(self):
        """
        Initialize the EnglishCharacterRemover class.
        This class is designed to remove English characters from text.
        """
        pass  # No specific attributes to initialize

    def remove_characters(self, line):
        """
        Removes all English alphabets (both lowercase and uppercase) from the input text.
        
        Parameters:
        line (str): The input text line from which English characters are to be removed.
        
        Returns:
        str: The input text with English characters removed.
        """
        return re.sub(r'[A-Za-z]', '', line)
    
# Class to remove empty lines
class EmptyLineRemover:
    """
    This class checks and removes empty lines after processing.
    """
    def is_empty_line(self, text):
        return len(text.strip()) == 0

class KannadaStemmer:
    def __init__(self):
        self.kannada_suffixes = sorted([
            'ಗಳು', 'ಕೆ', 'ಕ್ಕೆ', 'ಯನ್ನು', 'ನ್ನು', 'ಗಳನ್ನು', 'ದಿಂದ', 'ದ', 'ವು', 'ಇತ್ತು',
            'ತ್ತು', 'ಡಿದರು', 'ಡಿದ್ದ', 'ಅನು', 'ನ', 'ತೆ', 'ಯ', 'ವ', 'ಕ್ಕಿಂತ', 'ಹಾಗೂ',
            'ವಾಗಿ', 'ತಾನೆ', 'ವುದು', 'ತಿದೆ', 'ಬೇಕಿದೆ', 'ವಳು', 'ಇದರಿಂದ', 'ವನು', 'ವಳು',
            'ನಿಂದ', 'ದರೆ', 'ವರೆಗೆ', 'ಅಲ್ಲಿ', 'ಇಲ್ಲ', 'ತಿದೆ', 'ಅದಕ್ಕೆ', 'ಗಳ', 'ಹೇಗೆ',
            'ಇರುತ್ತದೆ', 'ವಿದ್ದ', 'ವುದು', 'ಕ್ಕಿಂತ', 'ಅಷ್ಟೇ', 'ಗೂ', 'ಅಷ್ಟು', 'ಇದೇ',
            'ಇಲ್ಲದೇ', 'ಇದಕ್ಕೆ', 'ನಲ್ಲಿ', 'ಮೇಲೆ', 'ಕೆಲಸ', 'ಇರುತ್ತದೆ', 'ಅಷ್ಟರಲ್ಲೂ',
            'ವಾದರೂ', 'ಗಿಂತ', 'ಇಲ್ಲದ', 'ದಷ್ಟು', 'ಮಾತ್ರ', 'ಸುತ್ತ', 'ಅವರ', 'ಬಗ್ಗೆ', 'ಇತರೆ',
            'ಮೇಲೆ', 'ಮಾಡಿದೆ', 'ದೊಡ್ಡ', 'ಹೆಚ್ಚಾಗಿ', 'ಅದರಿಂದ', 'ಮಾಡುತ್ತದೆ', 'ಕೇಳಿದ', 'ಎಷ್ಟು'
        ], key=len, reverse=True)

    # Function to stem a single word
    def stem(self, word):
        word = word.strip()
        for suffix in self.kannada_suffixes:
            if word.endswith(suffix):
                word = word[:-len(suffix)]
                return word

        return word

    # Function to apply stemming to a full sentence/phrase
    def stem_sentence(self, sentence):
        words = sentence.split()
        stemmed_words = [self.stem(word) for word in words]
        return ' '.join(stemmed_words)

class KannadaLemmatizer:
    """
    This class provides methods for lemmatizing Kannada words and sentences.
    """

    def __init__(self):
        # Initialize the suffixes as an instance variable
        self.kannada_suffixes = [
            'ಗಳು', 'ಕೆ', 'ಕ್ಕೆ', 'ಯನ್ನು', 'ನ್ನು', 'ಗಳಿಂದ', 'ದ', 'ವು', 'ಇತ್ತು',
            'ತನ್ಮೂಲಕ', 'ತಾರೆ', 'ಯಾಗ', 'ತಲೆ', 'ಯಲ್ಲಿ', 'ಮಾಡಿದೆ', 'ಮಾಡುತ್ತಿದೆ',
            'ವಿಲ್ಲ', 'ಬರುವ', 'ಗಮನ', 'ನಡುವೆ', 'ಹೊರಡಿದ', 'ಊಟ', 'ಮಾಡುವ', 'ಕೇಳು',
            'ನೀಡುವ', 'ಪಡುವ', 'ನಡೆಸುವ', 'ಎನ್ನು', 'ಸೀಮಿತ', 'ಹೋಗುವ',
            'ಬಿಡುವ', 'ಒಯ್ಯುವ', 'ಪರೀಕ್ಷಣೆ', 'ಗೊತ್ತಾಗುವುದು', 'ಸಾಧಿಸಲು', 'ಸಾಧನೆ',
            'ಮಾಡಲು', 'ಹೇಳಲು', 'ಹೇಳುವುದು'
        ]

    def kannada_lemmatize(self, word):
        """
        Lemmatize a single Kannada word based on context.
        """
        if re.match(r'.*?(ಮಾಡ|ಹೇಳ|ಊಟ|ಬಾಡಿ|ಹೋಗ|ಕಂಡ|ಎನ್ನು|ನಡ|ಸಾಧ|ಓದು).*$', word):
            if word.endswith('ತ್ತದೆ') or word.endswith('ನೀಡುತ್ತದೆ'):
                return re.sub(r'(.+?)(ತ್ತದೆ|ನೀಡುತ್ತದೆ)$', r'\1', word)
            elif word.endswith('ದೆ'):
                return re.sub(r'(.+?)(ದೆ)$', r'\1', word)
            elif word.endswith('ನ'):
                return re.sub(r'(.+?)(ನ)$', r'\1', word)

        if word.endswith('ಗಳು'):
            return re.sub(r'(.+?)(ಗಳು)$', r'\1', word)

        return word

    def lemmatize_sentence(self, sentence):
        """
        Lemmatize a sentence by processing each word.
        """
        words = sentence.split()
        lemmatized_words = [self.kannada_lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

class EnglishStemmer:
    def __init__(self):
        pass

    # Helper function to measure the "m" value (number of VC sequences)
    def measure(self, word):
        pattern = re.compile(r'[aeiou]+[^aeiou]+')
        matches = pattern.findall(word)
        return len(matches)

    # Helper function to check if a word contains a vowel
    def contains_vowel(self, word):
        return bool(re.search(r'[aeiou]', word))

    # Helper function to check double consonant at the end
    def ends_with_double_consonant(self, word):
        return bool(re.search(r'([^aeiou])\1$', word))

    # Helper function to check CVC (consonant-vowel-consonant) pattern
    def cvc(self, word):
        return bool(re.search(r'[^aeiou][aeiou][^aeiou]$', word)) and not re.search(r'[wxy]$', word)

    # Porter Stemmer algorithm
    def porter_stemmer(self, word):
        original_word = word
        if len(word) <= 2:
            return word  # Early exit for short words

        # Preserve common short words
        if word in {"hi", "do", "to", "it", "is", "be", "am", "are", "was", "were"}:
            return word

        # Step 1a: Plurals and participles
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[:-1]

        # Step 1b: "-ed" and "-ing" endings
        if word.endswith("eed"):
            if self.measure(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith("ed") and self.contains_vowel(word[:-2]):
            word = word[:-2]
            word = self.step_1b_helper(word)
        elif word.endswith("ing") and self.contains_vowel(word[:-3]):
            word = word[:-3]
            word = self.step_1b_helper(word)

        # Step 1c: "y" -> "i"
        if word.endswith("y") and self.contains_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2: Long suffixes
        suffixes_2 = {
            'ational': 'ate', 'tional': 'tion', 'enci': 'ence', 'anci': 'ance', 'izer': 'ize',
            'bli': 'ble', 'alli': 'al', 'entli': 'ent', 'eli': 'e', 'ousli': 'ous',
            'ization': 'ize', 'ation': 'ate', 'ator': 'ate', 'alism': 'al', 'iveness': 'ive',
            'fulness': 'ful', 'ousness': 'ous', 'aliti': 'al', 'iviti': 'ive', 'biliti': 'ble'
        }
        for suffix, replacement in suffixes_2.items():
            if word.endswith(suffix) and self.measure(word[:-len(suffix)]) > 0:
                word = word[:-len(suffix)] + replacement
                break

        # Step 3: "-icate", "-ative", "-alize"
        suffixes_3 = {'icate': 'ic', 'ative': '', 'alize': 'al', 'iciti': 'ic', 'ical': 'ic', 'ful': '', 'ness': ''}
        for suffix, replacement in suffixes_3.items():
            if word.endswith(suffix) and self.measure(word[:-len(suffix)]) > 0:
                word = word[:-len(suffix)] + replacement
                break

        # Step 4: "-al", "-ance", "-ence", etc.
        suffixes_4 = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize']
        for suffix in suffixes_4:
            if word.endswith(suffix) and self.measure(word[:-len(suffix)]) > 1:
                word = word[:-len(suffix)]
                break

        # Step 5a: "-e" removal
        if word.endswith("e"):
            if self.measure(word[:-1]) > 1 or (self.measure(word[:-1]) == 1 and not self.cvc(word[:-1])):
                word = word[:-1]

        # Step 5b: Double consonant at the end, remove one
        if self.ends_with_double_consonant(word) and word.endswith("l") and self.measure(word) > 1:
            word = word[:-1]

        return word

    # Step 1b helper (common after "-ed" or "-ing")
    def step_1b_helper(self, word):
        if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
            word = word + "e"
        elif self.ends_with_double_consonant(word) and not word.endswith(("l", "s", "z")):
            word = word[:-1]
        elif self.measure(word) == 1 and self.cvc(word):
            word = word + "e"
        return word

    # Function to stem a sentence
    def stem_sentence(self, sentence):
        words = sentence.split()
        stemmed_words = [self.porter_stemmer(word.lower()) for word in words]
        return ' '.join(stemmed_words)

class EnglishLemmatizer:
    """
    This class provides methods for lemmatizing English words and sentences.
    """

    def __init__(self):
        # Initialize the lemmatization map as an instance variable
        self.lemmatization_map = {
            'bought': 'buy',
            'went': 'go',
            'gone': 'go',
            'saw': 'see',
            'seen': 'see',
            'ran': 'run',
            'wrote': 'write',
            'written': 'write',
            'knew': 'know',
            'known': 'know',
            'had': 'have',
            'having': 'have',
            'did': 'do',
            'doing': 'do',
            'made': 'make',
            'getting': 'get',
            'got': 'get',
            'puts': 'put',
            'set': 'set',
            'took': 'take',
            'taken': 'take',
        }

    def custom_lemmatize(self, word):
        """
        Lemmatize a single English word based on common forms and irregular verbs.
        """
        # Handling common verb forms and irregular verbs
        if word in ['is', 'are', 'was', 'were']:
            return 'be'
        elif word in ['am']:
            return 'be'
        elif word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('es'):
            return word[:-2]
        elif word.endswith('s'):
            return word[:-1]

        # Handling common adjectives and adverbs
        if word.endswith('er'):
            return word[:-2]
        elif word.endswith('est'):
            return word[:-3]

        # Checking if the word is in the lemmatization map
        if word in self.lemmatization_map:
            return self.lemmatization_map[word]

        return word

    def lemmatize_sentence(self, sentence):
        """
        Lemmatize a sentence by processing each word.
        """
        if isinstance(sentence, str):
            words = sentence.split()
            lemmatized_words = [self.custom_lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        else:
            return ''

# Main class for complete text normalization
class TextNormalizerEnglish:
    """
    This class aggregates all the other preprocessing classes to provide
    a complete normalization pipeline.
    """
    def __init__(self, stopword_file=None, contraction_file=None):
        self.lowercase_handler = LowercaseHandler()
        self.whitespace_normalizer = WhitespaceNormalizer()
        self.punctuation_remover = PunctuationRemover()
        self.stopword_remover = StopwordRemover(stopword_file)
        self.empty_line_remover = EmptyLineRemover()
        self.contraction_handler = ContractionHandler(contraction_file)
        self.english_stemmer = EnglishStemmer()  # Initialize EnglishStemmer

    def normalize(self, text):
        """
        Perform complete text normalization.
        """
        # Step 1: Expand contractions
        text = self.lowercase_handler.to_lowercase(text)
        print(f"Lower case : {text}")
        # Step 2: Convert to lowercase
        text = self.contraction_handler.expand_contractions(text)
        print(f"Contaction Handler : {text}")
        # Step 3: Remove punctuation
        text = self.punctuation_remover.remove_punctuation(text)
        print(f"Punctuation Removal : {text}")
        # Step 4: Remove stopwords
        text = self.stopword_remover.remove_stopwords(text)
        print(f"Stop words : {text}")
        # Step 5: Stem the remaining words
        text = self.english_stemmer.stem_sentence(text)  # Stem the words after removing stopwords
        print(f"Stemmed Words : {text}")
        # Step 6: Normalize whitespaces
        text = self.whitespace_normalizer.normalize_whitespaces(text)
        print(f"Whitespace removal : {text}")
        # Step 7: Remove empty lines (if applicable)
        if self.empty_line_remover.is_empty_line(text):
            return ""
        
        return text

class TestPPEnglish:
    def __init__(self, input_file, output_file, stopword_file=None, contraction_file=None):
        """
        Initialize the TestPP class with input and output file paths.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.stopword_file = stopword_file
        self.contraction_file = contraction_file

    def process_english_file(self):
        """
        Processes the English text file by normalizing it using the TextNormalizer class.
        """
        normalizer = TextNormalizerEnglish(self.stopword_file, self.contraction_file)  # Initialize the TextNormalizer
        
        with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                normalized_line = normalizer.normalize(line)  # Normalize each line
                if normalized_line:  # Skip empty lines after normalization
                    outfile.write(normalized_line + "\n")  # Write the cleaned line to the output file

        print(f"Text processing completed. Processed file saved as: {self.output_file}")


class TextNormalizerKannada:
    """
    This class aggregates all the other preprocessing classes to provide
    a complete normalization pipeline for Kannada text.
    """
    def __init__(self):
        self.whitespace_normalizer = WhitespaceNormalizer()
        self.punctuation_remover = PunctuationRemover()
        self.stopword_remover = StopwordRemover()  # You can load a Kannada stopword file if you have one
        self.empty_line_remover = EmptyLineRemover()
        self.kannada_stemmer = KannadaStemmer()  # Assuming you have implemented this class already

    def normalize(self, text):
        """
        Perform complete text normalization for Kannada.
        """
        # Step 1: Convert to lowercase
        text = self.lowercase_handler.to_lowercase(text)

        # Step 2: Remove punctuation
        text = self.punctuation_remover.remove_punctuation(text)

        # Step 3: Remove stopwords
        text = self.stopword_remover.remove_stopwords(text)

        # Step 4: Normalize whitespaces
        text = self.whitespace_normalizer.normalize_whitespaces(text)

        # Step 5: Stem the text
        text = self.kannada_stemmer.stem_sentence(text)

        # Step 6: Remove empty lines (if applicable)
        if self.empty_line_remover.is_empty_line(text):
            return ""

        return text


class TestPPKannada:
    def __init__(self, input_file, output_file, stopword_file=None):
        """
        Initialize the TestPPKannada class with input and output file paths.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.stopword_file = stopword_file

    def process_kannada_file(self):
        """
        Processes the Kannada text file by normalizing it using the TextNormalizerKannada class.
        """
        normalizer = TextNormalizerKannada()  # Initialize the TextNormalizer for Kannada

        with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                normalized_line = normalizer.normalize(line)  # Normalize each line
                if normalized_line:  # Skip empty lines after normalization
                    outfile.write(normalized_line + "\n")  # Write the cleaned line to the output file

        print(f"Kannada text processing completed. Processed file saved as: {self.output_file}")
