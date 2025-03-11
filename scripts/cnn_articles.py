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

# Load dataset (use raw string for Windows paths)
file_path = 'cnn_articles.csv'
df = pd.read_csv(file_path)

# Handle missing values & duplicates
df = df.drop_duplicates()

# Convert text to lowercase
df['Title'] = df['Title'].astype(str).str.lower()
df['Content'] = df['Content'].astype(str).str.lower()

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
df['Title'] = df['Title'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['Content'] = df['Content'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove 'Summary' column if it exists
if 'Summary' in df.columns:
    df = df.drop(columns=['Summary'])

# Split concatenated words using wordninja
df['Title'] = df['Title'].apply(lambda x: ' '.join(wordninja.split(x)))
df['Content'] = df['Content'].apply(lambda x: ' '.join(wordninja.split(x)))

# Function to split merged words
def split_words(text):
    if isinstance(text, str):  # Ensure text is a string
        return " ".join(wordninja.split(text))
    return text  # Return as is if not a string
    df['Title'] = df['Title'].apply(split_words)
    df['Content'] = df['Content'].apply(split_words)

# Save cleaned dataset
cleaned_file_path = 'cnn_articles_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)

# Display cleaned dataset
print(df.head())