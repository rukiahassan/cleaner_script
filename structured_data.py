import pandas as pd
import urllib.parse
import os

# Function to load and clean the dataset
def load_and_clean_dataset(file_path):
    try:
        #load dataset
        dataset = pd.read_csv(file_path)

# Clean summary and content columns
        dataset['summary'] = dataset['summary'].apply(lambda x: ' '.join(eval(x)) if isinstance(eval(x), list) else x)
        dataset['content'] = dataset['content'].apply(lambda x: ' '.join(eval(x)) if isinstance(eval(x), list) else x)

# Replace 'nan' or empty lsits with None or an empty string
        dataset['summary'] = dataset['summary'].apply(lambda x: None if x == 'nan' or x == '[]' else x)
        dataset['content'] = dataset['content'].apply(lambda x: None if x == 'nan' or x == '[]' else x)
    
# Function to check fif the url is valid
        def is_valid_url(url):
            try:
                result = urllib.parse.urlparse(url)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        # Apply the URL Validation
        dataset ['url_valid'] = dataset['url'].apply(is_valid_url)

        # Structure the dataset
        structured_dataset = dataset[['title', 'summary', 'content', 'url', 'url_valid']]

        # Format the column names
        structured_dataset.columns = ['Title', 'Summary', 'Content', 'URL', 'Is URL Valid']

        return structured_dataset

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except  Exception as e:
        print(f"An error occurred: {e}")
        return None

# Get the current directory (Where the script is running)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Use the full path to the file (in the same directory as the script)
file_path = os.path.join(current_directory, 'baseball_cleaned.csv')
cleaned_dataset = load_and_clean_dataset(file_path)

# Check if the dataset is valid before using it 
if cleaned_dataset is not None:

    # Save the cleaned dataset to a new CSV file
    output_file_path = os.path.join(current_directory, 'structured_baseball_cleaned.csv')
    cleaned_dataset.to_csv(output_file_path, index=False)
    print(f"Dataset saved to {output_file_path}")
    print(cleaned_dataset.head())
else:
    print("The dataset is not valid.")


