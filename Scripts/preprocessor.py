import src.Preprocessing as PP
import src.BPE_tokenizer as tokenizer
import os

class PreProcess:
    def __init__(self):
        print("Loading the required files")
        self.english_contractions = './Data/english_contractions.json'
        if not os.path.isfile(self.english_contractions):
            raise FileNotFoundError("Contraction file does not exist")
        else:
            print("JSON file exists at location")

        # Initialize normalizers
        self.text_processor_eng = PP.TextNormalizerEnglish(self.english_contractions)
        self.text_processor_kan = PP.TextNormalizerKannada()

    def preprocess_english(self, sentence):
        print("Normalizing the English sentence")
        return self.text_processor_eng.normalize(sentence)

    def preprocess_kannada(self, sentence):
        print("Normalizing the Kannada sentence")
        return self.text_processor_kan.normalize(sentence)

preprocessor = PreProcess()
eng_text = preprocessor.preprocess_english("This is America")
kan_text = preprocessor.preprocess_kannada("ಇವರು ಸಂಶೋಧಕ ಸ್ವಭಾವದವರು.")
print(eng_text)
print(kan_text)