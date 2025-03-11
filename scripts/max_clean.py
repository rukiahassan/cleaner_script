import pandas as pd
import re
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
import unicodedata
import string

def clean_dataset(df):
    """
    Clean a dataset containing article data with columns:
    title, category, date, thumbnail, author, link, excerpt, and content
    
    Parameters:
    df (pandas.DataFrame): DataFrame with the columns to clean
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    import pandas as pd
    #from your_script_name import clean_dataset

    # Load your data
    df = pd.read_csv(r"C:\Users\Admin\Downloads\business_travel_page_8.csv")  # or any other data loading method


    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # 1. Clean titles
    cleaned_df['Title'] = cleaned_df['Title'].apply(clean_title)
    
    # 2. Clean categories
    cleaned_df['Category'] = cleaned_df['Category'].apply(clean_category)
    
    # 3. Clean and standardize dates
    cleaned_df['Date'] = cleaned_df['Date'].apply(clean_date)
    
    # 4. Clean thumbnail URLs
    cleaned_df['Thumbnail'] = cleaned_df['Thumbnail'].apply(clean_url)
    
    # 5. Clean author names
    cleaned_df['Author'] = cleaned_df['Author'].apply(clean_author)
    
    # 6. Clean link
    cleaned_df['Link'] = cleaned_df['Link'].apply(clean_url)
    
    # 7. Clean excerpt text
    cleaned_df['Excerpt'] = cleaned_df['Excerpt'].apply(clean_text)
    
    # 8. Clean content (most extensive cleaning)
    cleaned_df['Content'] = cleaned_df['Content'].apply(clean_content)
    
    # 9. Drop duplicates based on title and content
    cleaned_df.drop_duplicates(subset=['Title', 'Content'], inplace=True)
    
    # 10. Fill missing values with appropriate defaults
    cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df

def clean_title(title):
    """Clean and standardize article titles"""
    if pd.isna(title) or not isinstance(title, str):
        return ""
    
    # Remove extra whitespace
    title = " ".join(title.split())
    
    # Remove HTML tags if any
    title = BeautifulSoup(title, "html.parser").get_text()
    
    # Normalize unicode characters
    title = unicodedata.normalize('NFKC', title)
    
    # Capitalize title properly (keeping acronyms intact)
    title_words = title.split()
    
    small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                   'to', 'from', 'by', 'in', 'of', 'with', 'as'}
    
    for i, word in enumerate(title_words):
        # Always capitalize first and last word
        if i == 0 or i == len(title_words) - 1:
            # Check if it's an acronym (all caps)
            if not word.isupper():
                title_words[i] = word.capitalize()
        # Handle small words
        elif word.lower() in small_words:
            title_words[i] = word.lower()
        # Check if it's an acronym (all caps)
        elif not word.isupper():
            title_words[i] = word.capitalize()
    
    return " ".join(title_words)

def clean_category(category):
    """Clean and standardize categories"""
    if pd.isna(category) or not isinstance(category, str):
        return "Uncategorized"
    
    # Handle list-like categories (comma or pipe separated)
    if "," in category or "|" in category:
        categories = re.split(r',|\|', category)
        # Take the first category and clean it
        category = categories[0].strip()
    
    # Remove any special characters and extra spaces
    category = re.sub(r'[^\w\s]', '', category).strip()
    
    # Convert to title case
    category = category.title()
    
    return category

def clean_date(date_str):
    """Clean and standardize dates to ISO format (YYYY-MM-DD)"""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return pd.NaT
    
    # Remove any HTML tags if present
    date_str = BeautifulSoup(date_str, "html.parser").get_text().strip()
    
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%Y/%m/%d',                          # ISO-like formats
        '%d-%m-%Y', '%d/%m/%Y',                          # European formats
        '%m-%d-%Y', '%m/%d/%Y',                          # US formats
        '%B %d, %Y', '%b %d, %Y',                        # Month name formats
        '%d %B %Y', '%d %b %Y',                          # Day first with month name
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',      # ISO with time
        '%a, %d %b %Y %H:%M:%S %z',                      # RFC 2822 format (common in RSS)
    ]
    
    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, date_format)
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # If all formats fail, try to extract date using regex
    # Looking for patterns like YYYY-MM-DD, MM/DD/YYYY, etc.
    date_patterns = [
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # DD-MM-YYYY or MM-DD-YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            extracted_date = match.group(1)
            # Try the formats again with the extracted date
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(extracted_date, date_format)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    
    # If all else fails, return NaT (Not a Time)
    return pd.NaT

def clean_url(url):
    """Clean and validate URLs"""
    if pd.isna(url) or not isinstance(url, str):
        return ""
    
    # Remove extra whitespace
    url = url.strip()
    
    # Ensure URL has a scheme
    if url and not (url.startswith('http://') or url.startswith('https://')):
        url = 'https://' + url
    
    # Simple URL validation
    url_pattern = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if url and not url_pattern.match(url):
        return ""
    
    return url

def clean_author(author):
    """Clean and standardize author names"""
    if pd.isna(author) or not isinstance(author, str):
        return "Unknown"
    
    # Remove HTML tags if any
    author = BeautifulSoup(author, "html.parser").get_text()
    
    # Remove extra whitespace
    author = " ".join(author.split())
    
    # Remove titles and credentials (common patterns)
    author = re.sub(r'^(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+', '', author)
    author = re.sub(r',\s+(PhD|MD|JD|MBA|MA|BS|BA).*$', '', author)
    
    # Normalize unicode characters
    author = unicodedata.normalize('NFKC', author)
    
    # Proper capitalization for names
    if author and not author.isupper():  # Don't change if it's all caps (might be an acronym)
        name_parts = author.split()
        for i, part in enumerate(name_parts):
            # Check for hyphenated names
            if '-' in part:
                hyphen_parts = part.split('-')
                name_parts[i] = '-'.join(p.capitalize() for p in hyphen_parts)
            else:
                name_parts[i] = part.capitalize()
        
        author = " ".join(name_parts)
    
    return author

def clean_text(text):
    """Clean general text fields like excerpts"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Replace multiple periods with single ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    
    # Remove leading/trailing ellipsis
    text = re.sub(r'^\.\.\.', '', text)
    text = re.sub(r'\.\.\.$', '', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?])([^\s\d])', r'\1 \2', text)
    
    return text.strip()

def clean_content(content):
    """Clean main content text with more thorough processing"""
    if pd.isna(content) or not isinstance(content, str):
        return ""
    
    # Remove HTML tags but preserve some structure
    soup = BeautifulSoup(content, "html.parser")
    
    # Replace <br>, <p>, <div> tags with newlines for readability
    for tag in soup.find_all(['br', 'p', 'div']):
        tag.replace_with('\n' + tag.get_text() + '\n')
    
    # Get clean text
    text = soup.get_text()
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace within lines
    lines = text.split('\n')
    cleaned_lines = [" ".join(line.split()) for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove script content that might have been missed by BeautifulSoup
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
    
    # Remove any remaining HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Handle special characters and symbols
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?])([^\s\d])', r'\1 \2', text)
    
    return text.strip()

def fill_missing_values(df):
    """Fill missing values with appropriate defaults"""
    defaults = {
        'title': '',
        'category': 'Uncategorized',
        'date': pd.NaT,
        'thumbnail': '',
        'author': 'Unknown',
        'link': '',
        'excerpt': '',
        'content': ''
    }
    
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    
    return df

# Example usage
if __name__ == "__main__":
    # Load your dataset
    # df = pd.read_csv('your_dataset.csv')
    
    # For demonstration, create a small example dataset
    sample_data = {
        'title': ['Introduction to   DATA SCIENCE', '<b>Python Tips & Tricks</b>', None],
        'category': ['Data Science, Programming', 'Python|Coding', 'Tech'],
        'date': ['2023-01-15', '01/20/2023', 'January 25, 2023'],
        'thumbnail': ['http://example.com/img1.jpg', 'img2.jpg', None],
        'author': ['Dr. Jane Smith, PhD', 'john doe', None],
        'link': ['http://example.com/article1', 'example.com/article2', None],
        'excerpt': ['An intro to <i>data science</i> concepts...', 'Learn python tricks...', None],
        'content': ['<p>Data science is a field that...</p>', 'Python offers many useful tricks...', None]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Clean the dataset
    cleaned_df = clean_dataset(df)

    # Clean the dataset
    cleaned_df = clean_dataset(df)

    # Save the cleaned data
    cleaned_df.to_csv('cleaned_dataset.csv', index=False)
    
    
    # Display results
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)