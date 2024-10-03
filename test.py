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