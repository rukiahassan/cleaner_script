import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
from pathlib import Path
import os

def structure_dataset(df, output_dir=None):
    """
    Structure a cleaned dataset by validating types, creating derived features, 
    organizing categories, and preparing for analysis.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame with article data
    output_dir (str, optional): Directory to save structured data files
    
    Returns:
    pandas.DataFrame: Structured DataFrame with additional features
    dict: Summary statistics and metadata
    """
    import pandas as pd
from data_cleaning import clean_dataset
from data_structuring import structure_dataset

# First clean the data
df = pd.read_csv('cleaned_dataset.csv')
cleaned_df = clean_dataset(df)

# Then structure the cleaned data
structured_df, metadata = structure_dataset(cleaned_df, output_dir='structured_output')

# The structured_df now has additional features and proper organization
# The metadata contains useful statistics about your dataset
    
    
# Create a copy to avoid modifying the original
structured_df = df.copy()
    
# 1. Enforce proper data types
structured_df = enforce_data_types(structured_df)
    
# 2. Create derived features
structured_df = create_derived_features(structured_df)
    
# 3. Standardize and organize categories
structured_df = organize_categories(structured_df)
    
# 4. Validate data integrity
validate_data_integrity(structured_df)
    
# 5. Generate metadata and summary statistics
metadata = generate_metadata(structured_df)
    
# 6. Save structured data if output directory is provided
if output_dir:
   save_structured_data(structured_df, metadata, output_dir)
    
return structured_df, metadata

def enforce_data_types(df):
    """Enforce proper data types for each column"""
    # Create a copy to avoid modifying the original
    typed_df = df.copy()
    
    # Convert date to datetime
    typed_df['date'] = pd.to_datetime(typed_df['date'], errors='coerce')
    
    # Ensure text columns are strings
    text_columns = ['title', 'category', 'author', 'excerpt', 'content']
    for col in text_columns:
        if col in typed_df.columns:
            typed_df[col] = typed_df[col].astype(str)
            # Convert empty strings to NaN for consistency
            typed_df[col] = typed_df[col].replace('', np.nan)
            typed_df[col] = typed_df[col].replace('nan', np.nan)
    
    # Ensure URL columns are strings
    url_columns = ['thumbnail', 'link']
    for col in url_columns:
        if col in typed_df.columns:
            typed_df[col] = typed_df[col].astype(str)
            typed_df[col] = typed_df[col].replace('', np.nan)
            typed_df[col] = typed_df[col].replace('nan', np.nan)
    
    return typed_df

def create_derived_features(df):
    """Create new features derived from existing data"""
    # Create a copy to avoid modifying the original
    featured_df = df.copy()
    
    # 1. Extract year and month from date
    if 'date' in featured_df.columns:
        featured_df['year'] = featured_df['date'].dt.year
        featured_df['month'] = featured_df['date'].dt.month
        featured_df['day_of_week'] = featured_df['date'].dt.dayofweek
        
        # Create a year-month column for easier time series grouping
        featured_df['year_month'] = featured_df['date'].dt.to_period('M')
    
    # 2. Calculate content length features
    if 'content' in featured_df.columns:
        # Word count
        featured_df['word_count'] = featured_df['content'].fillna('').apply(lambda x: len(str(x).split()))
        
        # Paragraph count (assuming paragraphs are separated by newlines)
        featured_df['paragraph_count'] = featured_df['content'].fillna('').apply(
            lambda x: len([p for p in str(x).split('\n') if p.strip()])
        )
        
        # Content length category
        featured_df['content_length_category'] = pd.cut(
            featured_df['word_count'],
            bins=[0, 300, 800, 1500, float('inf')],
            labels=['Short', 'Medium', 'Long', 'Very Long']
        )
    
    # 3. Extract domain from link
    if 'link' in featured_df.columns:
        featured_df['domain'] = featured_df['link'].fillna('').apply(extract_domain)
    
    # 4. Create binary flags for data availability
    for col in ['thumbnail', 'author', 'excerpt']:
        if col in featured_df.columns:
            featured_df[f'has_{col}'] = (~featured_df[col].isna()).astype(int)
    
    return featured_df

def extract_domain(url):
    """Extract domain from URL"""
    if not isinstance(url, str) or not url:
        return np.nan
    
    try:
        # Simple regex to extract domain
        domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
        if domain_match:
            return domain_match.group(1)
    except:
        pass
    
    return np.nan

def organize_categories(df):
    """Standardize and organize categories"""
    # Create a copy to avoid modifying the original
    categorized_df = df.copy()
    
    if 'category' not in categorized_df.columns:
        return categorized_df
    
    # 1. Create a standardized category list
    category_mapping = {
        # Technology categories
        'technology': 'Technology',
        'tech': 'Technology',
        'programming': 'Technology/Programming',
        'coding': 'Technology/Programming',
        'software': 'Technology/Software',
        'hardware': 'Technology/Hardware',
        'ai': 'Technology/AI',
        'artificial intelligence': 'Technology/AI',
        'machine learning': 'Technology/AI',
        'data science': 'Technology/Data Science',
        'data': 'Technology/Data Science',
        'analytics': 'Technology/Data Science',
        'web development': 'Technology/Web Development',
        'web': 'Technology/Web Development',
        'mobile': 'Technology/Mobile',
        'apps': 'Technology/Mobile',
        'cloud': 'Technology/Cloud',
        'devops': 'Technology/DevOps',
        'security': 'Technology/Security',
        'cybersecurity': 'Technology/Security',
        
        # Business categories
        'business': 'Business',
        'finance': 'Business/Finance',
        'investing': 'Business/Finance',
        'marketing': 'Business/Marketing',
        'management': 'Business/Management',
        'entrepreneurship': 'Business/Entrepreneurship',
        'startup': 'Business/Entrepreneurship',
        'economy': 'Business/Economy',
        'careers': 'Business/Careers',
        'jobs': 'Business/Careers',
        
        # Science categories
        'science': 'Science',
        'physics': 'Science/Physics',
        'chemistry': 'Science/Chemistry',
        'biology': 'Science/Biology',
        'astronomy': 'Science/Astronomy',
        'space': 'Science/Astronomy',
        'environment': 'Science/Environment',
        'climate': 'Science/Environment',
        'medicine': 'Science/Medicine',
        'health': 'Science/Health',
        
        # Other common categories
        'news': 'News',
        'politics': 'Politics',
        'entertainment': 'Entertainment',
        'sports': 'Sports',
        'travel': 'Travel',
        'food': 'Food',
        'education': 'Education',
        'art': 'Art',
        'culture': 'Culture',
        'gaming': 'Gaming',
        'lifestyle': 'Lifestyle',
        'opinion': 'Opinion',
        'review': 'Review',
        'tutorial': 'Tutorial',
        'guide': 'Guide',
        'how-to': 'Guide',
        'interview': 'Interview',
    }
    
    # Function to map categories to standardized form
    def map_category(cat):
        # Handle NaN values
        if pd.isna(cat) or not cat:
            return 'Uncategorized'
        
        # Check for exact match
        cat_lower = cat.lower()
        if cat_lower in category_mapping:
            return category_mapping[cat_lower]
        
        # Check for partial matches
        for key, value in category_mapping.items():
            if key in cat_lower or cat_lower in key:
                return value
        
        # If no matches, return the original category
        return cat
    
    # Apply category mapping
    categorized_df['standardized_category'] = categorized_df['category'].apply(map_category)
    
    # 2. Create hierarchical categories
    categorized_df['category_level_1'] = categorized_df['standardized_category'].apply(
        lambda x: x.split('/')[0] if '/' in x else x
    )
    
    categorized_df['category_level_2'] = categorized_df['standardized_category'].apply(
        lambda x: x.split('/')[1] if '/' in x and len(x.split('/')) > 1 else np.nan
    )
    
    return categorized_df

def validate_data_integrity(df):
    """Validate data integrity and print warnings for issues"""
    # 1. Check for missing data in critical columns
    critical_cols = ['title', 'content']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"WARNING: {missing} rows have missing {col}")
    
    # 2. Check for future dates
    if 'date' in df.columns:
        future_dates = (df['date'] > datetime.now()).sum()
        if future_dates > 0:
            print(f"WARNING: {future_dates} rows have future dates")
    
    # 3. Check for very short content
    if 'word_count' in df.columns:
        short_content = (df['word_count'] < 50).sum()
        if short_content > 0:
            print(f"WARNING: {short_content} rows have very short content (< 50 words)")
    
    # 4. Check for duplicate titles
    if 'title' in df.columns:
        duplicates = df['title'].duplicated().sum()
        if duplicates > 0:
            print(f"WARNING: {duplicates} duplicate titles found")
    
    # 5. Check for invalid URLs
    for url_col in ['thumbnail', 'link']:
        if url_col in df.columns:
            invalid_urls = df[url_col].apply(
                lambda x: False if pd.isna(x) else not bool(re.match(r'^https?://', str(x)))
            ).sum()
            if invalid_urls > 0:
                print(f"WARNING: {invalid_urls} rows have invalid {url_col} URLs")

def generate_metadata(df):
    """Generate metadata and summary statistics for the dataset"""
    metadata = {
        'dataset_info': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'row_count': len(df),
            'column_count': len(df.columns),
        },
        'column_stats': {},
        'category_distribution': {},
        'time_distribution': {},
        'content_stats': {}
    }
    
    # 1. Generate column statistics
    for col in df.columns:
        # Skip derived columns like year, month
        if col in ['year', 'month', 'day_of_week', 'year_month', 
                  'word_count', 'paragraph_count', 'has_thumbnail', 
                  'has_author', 'has_excerpt', 'domain']:
            continue
            
        col_stats = {
            'missing_count': df[col].isna().sum(),
            'missing_percentage': round(df[col].isna().mean() * 100, 2),
        }
        
        # Add statistics based on column type
        if df[col].dtype == 'object':
            col_stats['unique_count'] = df[col].nunique()
            
            # For text columns, add length statistics
            if col in ['title', 'excerpt', 'content']:
                col_stats['avg_length'] = df[col].fillna('').apply(len).mean()
                col_stats['max_length'] = df[col].fillna('').apply(len).max()
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_stats['min'] = df[col].min()
            col_stats['max'] = df[col].max()
            col_stats['mean'] = df[col].mean()
            col_stats['median'] = df[col].median()
            
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_stats['min_date'] = df[col].min().strftime('%Y-%m-%d') if not pd.isna(df[col].min()) else None
            col_stats['max_date'] = df[col].max().strftime('%Y-%m-%d') if not pd.isna(df[col].max()) else None
            col_stats['date_range_days'] = (df[col].max() - df[col].min()).days if not pd.isna(df[col].min()) else None
        
        metadata['column_stats'][col] = col_stats
    
    # 2. Generate category distribution
    if 'standardized_category' in df.columns:
        category_counts = df['standardized_category'].value_counts().to_dict()
        metadata['category_distribution'] = {k: v for k, v in sorted(
            category_counts.items(), key=lambda item: item[1], reverse=True
        )}
    
    # 3. Generate time distribution
    if 'year_month' in df.columns:
        time_counts = df['year_month'].value_counts().sort_index()
        metadata['time_distribution'] = {str(k): v for k, v in time_counts.items()}
    
    # 4. Generate content statistics
    if 'word_count' in df.columns:
        metadata['content_stats'] = {
            'avg_word_count': df['word_count'].mean(),
            'median_word_count': df['word_count'].median(),
            'max_word_count': df['word_count'].max(),
            'min_word_count': df['word_count'].min(),
            'word_count_distribution': {
                'short': (df['word_count'] < 300).sum(),
                'medium': ((df['word_count'] >= 300) & (df['word_count'] < 800)).sum(),
                'long': ((df['word_count'] >= 800) & (df['word_count'] < 1500)).sum(),
                'very_long': (df['word_count'] >= 1500).sum()
            }
        }
        
        if 'paragraph_count' in df.columns:
            metadata['content_stats']['avg_paragraph_count'] = df['paragraph_count'].mean()
            metadata['content_stats']['avg_words_per_paragraph'] = (
                df['word_count'] / df['paragraph_count'].replace(0, 1)
            ).mean()
    
    return metadata

def save_structured_data(df, metadata, output_dir):
    """Save structured data and metadata to files"""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main dataframe
    df.to_csv(output_path / 'structured_data.csv', index=False)
    
    # Save as SQLite database
    import sqlite3
    conn = sqlite3.connect(output_path / 'structured_data.db')
    df.to_sql('articles', conn, if_exists='replace', index=False)
    conn.close()
    
    # Save metadata as JSON
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create category subsets
    if 'category_level_1' in df.columns:
        for category in df['category_level_1'].unique():
            if pd.isna(category):
                continue
                
            category_df = df[df['category_level_1'] == category]
            
            # Only save if there are enough rows
            if len(category_df) > 5:
                clean_category = re.sub(r'[^\w]', '_', category)
                category_df.to_csv(output_path / f'category_{clean_category}.csv', index=False)
    
    # Create time period subsets
    if 'year' in df.columns:
        for year in df['year'].unique():
            if pd.isna(year):
                continue
                
            year_df = df[df['year'] == year]
            
            # Only save if there are enough rows
            if len(year_df) > 5:
                year_df.to_csv(output_path / f'year_{int(year)}.csv', index=False)
    
    print(f"Structured data saved to {output_dir}")
    print(f"Files created:")
    print(f"  - structured_data.csv (main dataset)")
    print(f"  - structured_data.db (SQLite database)")
    print(f"  - metadata.json (dataset metadata)")
    print(f"  - category_*.csv files (category subsets)")
    print(f"  - year_*.csv files (yearly subsets)")

# Example usage
if __name__ == "__main__":
    # This would typically come after running the cleaning script
    # from data_cleaning import clean_dataset
    
    # For demonstration, create a sample dataframe that would result from cleaning
    sample_data = {
        'title': ['Introduction to Data Science', 'Python Tips & Tricks', 'Machine Learning Basics'],
        'category': ['Data Science', 'Python', 'Machine Learning'],
        'date': ['2023-01-15', '2023-01-20', '2023-01-25'],
        'thumbnail': ['https://example.com/img1.jpg', 'https://example.com/img2.jpg', 'https://example.com/img3.jpg'],
        'author': ['Jane Smith', 'John Doe', 'Alice Johnson'],
        'link': ['https://example.com/article1', 'https://example.com/article2', 'https://example.com/article3'],
        'excerpt': ['An intro to data science concepts...', 'Learn python tricks...', 'Basics of machine learning...'],
        'content': ['Data science is a field that...\n\nIt involves many techniques.', 
                   'Python offers many useful tricks...\n\nHere are some examples.', 
                   'Machine learning is a subset of AI...\n\nIt includes many algorithms.']
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Apply the structuring script
    structured_df, metadata = structure_dataset(sample_df, output_dir='structured_data')
    
    # Display results
    print("\nStructured DataFrame:")
    print(structured_df.head())
    print("\nMetadata Sample:")
    print(json.dumps(metadata, indent=2))