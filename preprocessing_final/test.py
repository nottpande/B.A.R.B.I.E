# Specify file paths for input and output
from preprocessing_final.Preprocessing import TestPPEnglish


input_file = "manual.txt"              # Input file containing English sentences
output_file = "manual_o.txt"           # Output file for storing the cleaned and preprocessed text
stopword_file = "english_stopwords.txt"  # Optional stopword file (if available)
contraction_file = "english_contractions.json"   # Optional contraction file

# Create an instance of the TestPP class and run the processing function
text_processor = TestPPEnglish(input_file, output_file, stopword_file, contraction_file)
text_processor.process_english_file()