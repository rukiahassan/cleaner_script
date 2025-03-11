import pandas as pd
import re
import nltk
import wordninja
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset (use raw string for Windows paths)
file_path = r'C:\Workspace\Simba\Data\healthcare\healthcare_data.csv'
df = pd.read_csv(file_path)

# Handle missing values & duplicates
df = df.drop_duplicates()

# Convert text to lowercase
df['summary'] = df['summary'].astype(str).str.lower()
df['content'] = df['content'].astype(str).str.lower()

# Remove special characters except spaces
df['summary'] = df['summary'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['content'] = df['content'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove 'links' column if it exists
if 'links' in df.columns:
    df = df.drop(columns=['links'])

# Split concatenated words using wordninja
df['summary'] = df['summary'].apply(lambda x: ' '.join(wordninja.split(x)))
df['content'] = df['content'].apply(lambda x: ' '.join(wordninja.split(x)))

# Tokenize the cleaned text
df['summary'] = df['summary'].apply(lambda x: word_tokenize(x))
df['content'] = df['content'].apply(lambda x: word_tokenize(x))

# Save cleaned dataset
cleaned_file_path = r'C:\Workspace\Simba\Data\healthcare\cleaned_healthcare_data.csv'
df.to_csv(cleaned_file_path, index=False)

# Display cleaned dataset
print(df.head())


