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
file_path = 'data\cnn_articles.csv'
df = pd.read_csv(file_path)

# Handle missing values & duplicates
df = df.drop_duplicates()

# check duplicates
df.isnull().sum()

# Display cleaned dataset
print(df.head())

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

# Remove 'summary' column if it exists
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

 #split wordS
df['Title'] = df['Title'].apply(split_words)
df['Content'] = df['Content'].apply(split_words)

#drop duplicates
df = df.drop_duplicates()

#dropping null values
df.dropna(inplace = True)
df.info()

#dropping Unnecesary column
df.drop(columns=['links'],inplace=True)

#check for whitespaces by checking length of values
df['summary_length'] = df['summary'].apply(lambda x: len(str(x)))
print(df[['summary', 'summary_length']].sort_values('summary_length').head(10))

#converting whitespaces to NaN
df['summary'] = df['summary'].apply(lambda x: np.nan if isinstance(x, str) and x.strip()=='' else x)

#converting whitespaces to NaN
df['content'] = df['content'].apply(lambda x: np.nan if isinstance(x, str) and x.strip()=='' else x)

df.isna().sum()
#replace nan with unknown
df['summary'] = df['summary'].replace(np.nan, 'unknown')

# Reset the index and drop the old index
df.reset_index(drop=True, inplace=True)
#remove urls
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

df['summary'] = df['summary'].apply(remove_url)
df['content'] = df['content'].apply(remove_url)

#This function removes punctuations
def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['summary'] = df['summary'].apply(remove_punct)
df['content'] = df['content'].apply(remove_punct)

#This function removes username
def remove_at(text):
    return re.sub(r'@\w+', '', text)

df['summary'] = df['summary'].apply(remove_at)
df['content'] = df['content'].apply(remove_at)

#This function removes html tags
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

df['summary'] = df['summary'].apply(remove_html)
df['content'] = df['content'].apply(remove_html)

#pip install wordninja
import wordninja

#drop unnecessary columns
df.drop(columns=['summary_length','content_length'],inplace=True)

#Viewing the content column
df['content'].tolist()

def clean_text(text, politics_terms=None):
    """
    Cleans text while considering cases where the text may contain code snippets.
    """
    if not isinstance(text, str):
        return text

    if is_code(text):
        return text  # Do not alter code snippets

    text = text.lower().replace("\n", " ")

    # Refined CamelCase splitting
    text = split_camel_case(text, politics_terms_terms)

    # Combined regex operations with pre-compiled patterns
    text = non_ascii_re.sub(" ", text)  # Remove non-ASCII
    text = number_word_re.sub(r"\1 \2", text)  # Separate numbers from words
    text = extra_space_re.sub(" ", text).strip()  # Remove extra spaces
    text = single_char_re.sub("", text)  # Remove single characters only

    text = process_numbers(text)
    text = process_math_expressions(text)

    return text.strip()

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


# Save cleaned dataset

df.to_csv('cnn_articles_cleaned.csv', index=False)

