# examples/example.py
import sys
import os
import re  # Import the regular expression module
import pandas as pd
#from data_cleaner.core import DataCleaner  # Import the class


# Add the data_cleaner package to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




# --- New Function  ---
def standardize_price_column_if_needed(df, column_name, currency_symbols=['$', 'USD', '€', '£']):
    """
    Dynamically analyzes a column and standardizes it if it appears to contain price data.


    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to analyze and potentially standardize.
    currency_symbols : list, optional
        A list of currency symbols and keywords to look for. Defaults to ['$','USD', '€', '£'].
  

    Returns:
    -------
    pandas.DataFrame
        The modified DataFrame (standardized if price-like data is detected).
    """
    logging.info(f"Analyzing column '{column_name}' for potential price data...")


    # 1. Check if the column exists
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' not found in DataFrame.")
        return df


    # 2. Check the data type of the column
    if not pd.api.types.is_object_dtype(df[column_name].dtype) and not pd.api.types.is_numeric_dtype(df[column_name].dtype):
        logging.info(f"Column '{column_name}' is not of type object (string) or numeric. Skipping standardization.")
        return df  # Skip if not string or number


    # 3. Define the data type cleaning
    # Convert any numeric data into string
    df[column_name] = df[column_name].astype(str)


    # 4. Check for potential currency symbols and patterns.
    def contains_currency(value, currency_symbols):
        """Check if a value contains currency symbols or common price patterns."""
        if pd.isnull(value):
            return False


        # Check currency symbols from data
        for symbol in currency_symbols:
            if symbol in str(value):
                return True
        # Check for a common price patterns (e.g., number with commas or decimals)
        pattern = r'^[\\$]?\\d+[\\.,]?\\d*[\\$]?$'
        if re.match(pattern, str(value)):
            return True


        return False #return false if the data is not found
    #Check if this column matches
    has_currency_data = df[column_name].apply(lambda x: contains_currency(x, currency_symbols)).any() # Check ANY
    if not has_currency_data:
        logging.info(f"Column '{column_name}' does not appear to contain price data. Skipping standardization.")
        return df #return nothing if there is no currency data


    # 5. Standardize the Price (call function from Part 1)
    logging.info(f"Column '{column_name}' appears to contain price data. Proceeding with standardization...")


    # 6. Create function for cleaning
    def clean_price(price):
        """
        Clean price value from string format to numerical


        Parameters:
        -----------
        price : str, int, float
            Price value in USD format


        Returns:
        --------
        float
            Standardized price value as float number
        """
        try:
            if isinstance(price, (int, float)):
                return float(price)


            # Remove currency symbol, commas, and whitespace
            if isinstance(price, str):
                price = price.strip()
                price = price.replace('$', '')
                price = price.replace(',', '')
                price = price.replace('USD', '')
                price = price.replace(' ', '')
                price = price.replace('€', '')
                price = price.replace('£', '')


                # Convert to float
                return float(price)


            return None  # Returns None for non-string and non-numeric values


        except (ValueError, TypeError):
            return None #returns None


    # Apply the cleaning function and handle any errors
    cleaned_prices = df[column_name].apply(clean_price)


    # Log information about any failed conversions
    null_count = cleaned_prices.isnull().sum()
    if null_count > 0:
        logging.warning(f"Warning: {null_count} price values could not be converted")


    df[column_name] = cleaned_prices  # Replace the column with cleaned prices
    return df
# --- End of new Fucntion


# --- Test Function ---
def test_standardize_price_column_if_needed():
    # Test cases
    test_data = {
        'valid_prices': ['$100', '1,200 USD', 150.50, '50', None, 'Invalid'],
        'mixed_data': ['Text', 123, '$456', '789'],
        'no_currency': ['Item A', 'Item B', 'Item C', 'Item D'],
        'numeric': [100, 200, 300, 400]
    }
    df = pd.DataFrame(test_data)


    # Testing for column 1 (column_name)
    analyzed_df = standardize_price_column_if_needed(df.copy(), "valid_prices")
    # Verify if the cleaned price matches values
    assert analyzed_df["valid_prices"].tolist() == [100.0, 1200.0, 150.5, 50.0, None, None]


    analyzed_df = standardize_price_column_if_needed(df.copy(), "mixed_data")
    # Verify if the cleaned price matches values
    assert analyzed_df["mixed_data"].tolist() == ['Text', 123.0, 456.0, '789']


    analyzed_df = standardize_price_column_if_needed(df.copy(), "no_currency")
    # Verify if the cleaned price matches values
    assert analyzed_df["no_currency"].tolist() ==  ['Item A', 'Item B', 'Item C', 'Item D']
    """
    Standardize price values from USD format to numerical values


    Parameters:
    -----------
    price_series : pandas.Series
        Series containing price values in USD format


    Returns:
    --------
    pandas.Series
        Standardized price values as float numbers
    """
    import logging
    def clean_price(price):
        try:
            if isinstance(price, (int, float)):
                return float(price)


            # Remove USD symbol, commas, and whitespace
            if isinstance(price, str):
                price = price.strip()
                price = price.replace('$', '')
                price = price.replace(',', '')
                price = price.replace('USD', '')
                price = price.replace(' ', '')


                # Convert to float
                return float(price)


            return None  # Returns None for non-string and non-numeric values


        except (ValueError, TypeError):
            return None #returns None


    # Apply the cleaning function and handle any errors
    cleaned_prices = price_series.apply(clean_price)


    # Log information about any failed conversions
    null_count = cleaned_prices.isnull().sum()
    if null_count > 0:
        logging.warning(f"Warning: {null_count} price values could not be converted")


    return cleaned_prices
# --- End of Test function ---


# --- Data Loading and Usage ---
def run_data_cleaning(input_file, output_file, price_column=None, config=None):
    """
    Loads data, applies cleaning, tests function, and saves the results.


    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
        price_column (str): (optional) The price column to standardize
        config (dict, optional): Configuration dictionary. Defaults to None.
    """


    try:
        # 1. Instantiate the DataCleaner
        cleaner = DataCleaner(config)


        # 2. Load your data
        cleaner.load_data(input_file)


        # 3. Test the new function (if price_column is given)
        if price_column:
            print(f"\n--- Testing standardize_price on column '{price_column}' ---")
            original_prices = cleaner.df[price_column].copy() #copy before changing
            cleaned_prices = standardize_price(cleaner.df[price_column])
            print("Original Prices:\n", original_prices)
            print("Cleaned Prices:\n", cleaned_prices)


            # Add some basic checks (you'll want to customize these)
            num_nulls = cleaned_prices.isnull().sum()
            print(f"\nNumber of Null Values After Cleaning: {num_nulls}") #Check data quality after transformation


            # Optional: Replace the column with the standardized prices
            cleaner.df[price_column] = cleaned_prices


            print("DataFrame with Cleaned Prices:\n", cleaner.df.head()) # Show the first few rows of the dataframe with the tested column.
            print("You should manually check the results above to ensure correctness!\n")


        # 4. Apply the cleaning pipeline
        cleaner.apply_cleaning_pipeline()


        # 5. Save the cleaned data
        cleaner.save_cleaned_data(output_file, encoding='utf-8')
        print(f"Cleaned data saved successfully to {output_file}")


        # Optional: Save other artifacts (flagged records, validation errors)
        cleaner.save_flagged_records(output_file.replace(".csv", "_flagged.csv"), encoding='utf-8')
        cleaner.save_validation_errors(output_file.replace(".csv", "_validation_errors.csv"), encoding='utf-8')


    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


# --- Example Usage ---
if __name__ == "__main__":
    # Test 1: For function
    # 1. Create a sample DataFrame
    data = {'Price': ['$100', '1,200 USD', 150.50, '50', None, 'Invalid'], 'Description': ['item1', 'item2', 'item3', 'item4', 'item5', 'item6']}
    test_df = pd.DataFrame(data)


    # 2. Call the function
    #test_standardize_price_column_if_needed() # Testing Function
    # 3. Testing with dataset to see transformations
    #transformed_df = standardize_price_column_if_needed(test_df.copy(), 'Price')


    # 4. Print results
    print("Original DataFrame:\n", test_df.to_string())
    print("\nModified DataFrame:\n", transformed_df.to_string())










   # Test 2: Add it to dataset


    # Load dataset
    # 1. Define file paths
    input_csv_file = "data/javascript_tutorial_data.csv"  # Relative to project root
    output_csv_file = "output/cleaned_javascript_tutorial_data.csv"  # Relative to project root
    test_price_column = "Price" #Name of the column to test the standardize_price function


    # 2. Optional configuration (you can also load from a file)
    config = {
        "iqr_threshold": 1.5,  # Example configuration
        "missing_value_handling": {
            "numerical": "median",
            "categorical": "most_frequent"
        }
    }


    # 3. Ensure 'data' and 'output' directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)


    # 4. Create a sample data file if it doesn't exist.
    if not os.path.exists(input_csv_file):
       
        print(f"Created a dummy data file at {input_csv_file}")
        dummy_data = {'col1': [1, 2, None, 4], 'col2': ['A', 'B', 'C', None],
                      'Price': ['$100', '1,200 USD', 150.50, '50', None, 'Invalid']}
        df = pd.DataFrame(dummy_data)
        df.to_csv(input_csv_file, index=False)
        print(f"Created a dummy data file at {input_csv_file}")
    df = pd.read_csv(input_csv_file)
    # 3. Add and try to clean it if there is price in the column
    test_standardize_price_column_if_needed() # Testing Function
    print(df.to_string())
    # 5. Run the data cleaning process
    run_data_cleaning(input_csv_file, output_csv_file, test_price_column, config) # pass the price_column.

    #  how to tokenization
    def tokenize_text(df, summary_col='summary', content_col='content'):
    
    Tokenizes the text in the specified columns (summary and content) of the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'summary' and 'content' columns
    summary_col (str): Name of the summary column to tokenize (default is 'summary')
    content_col (str): Name of the content column to tokenize (default is 'content')
    
    Returns:
    pandas.DataFrame: DataFrame with tokenized columns added
    
    # Tokenize the 'summary' column at the word level and sentence level
    df[f'{summary_col}_word_tokens'] = df[summary_col].apply(word_tokenize)
    df[f'{summary_col}_sentence_tokens'] = df[summary_col].apply(sent_tokenize)

    # Tokenize the 'content' column at the word level and sentence level
    df[f'{content_col}_word_tokens'] = df[content_col].apply(word_tokenize)
    df[f'{content_col}_sentence_tokens'] = df[content_col].apply(sent_tokenize)

    return df


     # remove emojis
     import pandas as pd
     import re

def remove_emojis(text):
    """
    Removes emojis from the text using a regex pattern.
    """
    if isinstance(text, str):
        return re.sub(r'[^\w\s,.-]', '', text)  # Removes non-alphanumeric characters (emojis included)
    return text
    # Remove emojis from 'Summary' and 'Content' columns
df['Summary'] = df['Summary'].apply(remove_emojis)
df['Content'] = df['Content'].apply(remove_emojis)

print(df)



             

pip install SpeechRecognition moviepy
 
import speech_recognition as sr
from moviepy.editor import AudioFileClip
 
# Extract audio from video

video_path = 'clip.mp4'

audio_path = 'audio.wav'

video = AudioFileClip(video_path)

video.audio.write_audiofile(audio_path)
 
# Initialize recognizer

recognizer = sr.Recognizer()
 
# Load audio file and transcribe itwith sr.AudioFile(audio_path) as source:

    audio_data = recognizer.record(source)

    text = recognizer.recognize_google(audio_data)
print(text)
 
 # Remove leading zeros in all numeric columns (if any)for column in dataset.select_dtypes(include=['object']).columns:
            dataset[column] = dataset[column].apply(lambda x: x.lstrip('0') ifisinstance(x, str) else x)