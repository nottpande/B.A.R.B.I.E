import pandas as pd

try:
    print("Performing data extraction...")
    try:
        print("Extracting the english sentences...")
        with open('english.txt', 'r', encoding='utf-8') as file:
            eng_sentence = file.readlines()
            eng_sentence = [sentence.strip() for sentence in eng_sentence if sentence.strip()]
        print("English text extracted successfully!")
    except Exception as e:
        print("Faced an error while extracting English Sentences")
        print(e)
    
    try:
        print("Extracting the Kannada sentences...")
        with open('kannada.txt', 'r', encoding='utf-8') as file:
            kan_sentence = file.readlines()
            kan_sentence = [sentence.strip() for sentence in kan_sentence if sentence.strip()]
        print("Kannada text extracted successfully!")
    except Exception as e:
        print("Faced an error while extracting Kannada Sentences")
        print(e)

    try:
        print("Combining the texts to generate the dataset...")
        if len(eng_sentence) == len(kan_sentence):
            df = pd.DataFrame({
                'English': eng_sentence,
                'Kannada': kan_sentence
    })
    except Exception as e:
        print("Faced an error while creating the dataset")
        print(e)

    try:
        print("Saving the Dataset...")
        df.to_csv('dataset_kag.csv',index = False)
    except Exception as e:
        print("Faced an error while saving the dataset")
        print(e)
    
    print("Dataset created successfully!!!")
    
except Exception as e:
        print("Faced an error while saving the dataset")
        print(e)