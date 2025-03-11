import pandas as pd
import re
import nltk
import wordninja
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

#%pip install pyspellchecker
from spellchecker import SpellChecker

def check_spelling(text):
    # Initialize the spell checker
    spell = SpellChecker()
             
    # Split the input text into words
    words = text.split()
    
    # Find words that may be misspelled
    misspelled = spell.unknown(words)
    
    corrections = {}
    for word in misspelled:
        # Get the most likely correction
        corrections[word] = spell.correction(word)
    
    return corrections


# Load dataset (use raw string for Windows paths)
file_path = 'data\human_body_data.csv'
df = pd.read_csv(file_path)

# Handle missing values & duplicates
df = df.drop_duplicates()

# Convert text to lowercase
df['summary'] = df['summary'].astype(str).str.lower()
df['content'] = df['content'].astype(str).str.lower()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer() 

# Function to lemmatize text
def lemmatize_text(text):
    if isinstance(text, str):  # Ensure text is a string
        words = text.split()
        return " ".join(lemmatizer.lemmatize(word) for word in words)
    return text  # Return as is if not a string

#Download necessary NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Remove Stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Remove special characters except spaces
df['summary'] = df['summary'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['content'] = df['content'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove 'links' column if it exists
if 'links' in df.columns:
    df = df.drop(columns=['links'])

# Split concatenated words using wordninja
df['summary'] = df['summary'].apply(lambda x: ' '.join(wordninja.split(x)))
df['content'] = df['content'].apply(lambda x: ' '.join(wordninja.split(x)))

# Function to split merged words
def split_words(text):
    if isinstance(text, str):  # Ensure text is a string
        return " ".join(wordninja.split(text))
    return text  # Return as is if not a string
    df['summary'] = df['summary'].apply(split_words)
    df['content'] = df['content'].apply(split_words)

# Save cleaned dataset
cleaned_file_path = 'human_body_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)

# Display cleaned dataset
print(df.head())