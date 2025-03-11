#pip install pyspellchecker
#pip install nltk
import nltk
nltk.download('punkt')


# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


import pandas as pd
import re
import nltk
import wordninja
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


import pandas as pd

def clean_dataset(file_path, output_path=None):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Display initial dataset info
    print("Initial dataset info:")
    print(df.info())
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Standardize column names (lowercase and replace spaces with underscores)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.summary
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')  # Forward fill, then backward fill
    
    # Convert data types if necessary
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # Keep as object if conversion fails
    
    # Trim whitespace from string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Display cleaned dataset info
    print("Cleaned dataset info:")
    print(df.info())
    
    # Save cleaned dataset
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")
    
    return df

# Example usage
if __name__ == "__main__":
    input_file = "your_dataset.csv"  # Replace with your dataset file
    output_file = "cleaned_dataset.csv"
    clean_dataset(input_file, output_file)
