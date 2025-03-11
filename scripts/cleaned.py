import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def clean_dataset(file_path, output_path=None):
    """
    Clean a dataset by handling missing values, duplicates, outliers, and standardizing formats.
    
    Parameters:
    -----------
    file_path : str
        Path to the input dataset file (CSV, Excel, etc.)
    output_path : str, optional
        Path where the cleaned dataset will be saved. If None, uses '[original_name]_cleaned.[ext]'
    
    Returns:
    --------
    pd.DataFrame
        The cleaned dataframe
    """
    # Determine file type and read accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Original dataset shape: {df.shape}")
    
    # Make a copy to preserve original data
    df_cleaned = df.copy()
    
    # Basic information and missing values
    print("\n--- Dataset Overview ---")
    print(f"Number of rows: {df_cleaned.shape[0]}")
    print(f"Number of columns: {df_cleaned.shape[1]}")
    print("\n--- Missing Values Summary ---")
    missing_values = df_cleaned.isnull().sum()
    missing_percent = (missing_values / len(df_cleaned) * 100).round(2)
    missing_table = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print(missing_table[missing_table['Missing Values'] > 0])
    
    # 1. Remove duplicate rows
    duplicate_count = df_cleaned.duplicated().sum()
    if duplicate_count > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"\nRemoved {duplicate_count} duplicate rows.")
    
    # 2. Handle missing values (column-specific handling)
    for column in df_cleaned.columns:
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(df_cleaned[column]):
            # Fill with median for numeric data (more robust than mean against outliers)
            if df_cleaned[column].isnull().sum() > 0:
                median_value = df_cleaned[column].median()
                df_cleaned[column].fillna(median_value, inplace=True)
                print(f"Filled missing values in '{column}' with median: {median_value}")
        
        # Handle categorical columns
        elif pd.api.types.is_object_dtype(df_cleaned[column]) or pd.api.types.is_categorical_dtype(df_cleaned[column]):
            # Fill with mode for categorical data
            if df_cleaned[column].isnull().sum() > 0:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in '{column}' with mode: {mode_value}")
        
        # Handle datetime columns
        elif pd.api.types.is_datetime64_dtype(df_cleaned[column]):
            # Forward fill for time series data
            if df_cleaned[column].isnull().sum() > 0:
                df_cleaned[column].fillna(method='ffill', inplace=True)
                print(f"Forward-filled missing values in datetime column '{column}'")
    
    # 3. Convert data types where needed
    for column in df_cleaned.columns:
        # Try to convert string columns that might be numeric
        if pd.api.types.is_object_dtype(df_cleaned[column]):
            try:
                # Check if this column appears to be numeric but stored as string
                if df_cleaned[column].str.match(r'^-?\d+\.?\d*$').dropna().all():
                    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
                    print(f"Converted '{column}' to numeric type")
            except:
                # Keep as is if conversion fails
                pass
            
            # Check for potential dates
            try:
                if df_cleaned[column].str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').dropna().all():
                    df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
                    print(f"Converted '{column}' to datetime type")
            except:
                pass
    
    # 4. Handle outliers in numeric columns (using IQR method)
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df_cleaned[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)][column]
        if len(outliers) > 0:
            print(f"\nFound {len(outliers)} outliers in '{column}'")
            print(f"Range: {outliers.min()} to {outliers.max()}")
            
            # Cap outliers rather than removing them
            df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"Capped outliers in '{column}' to range: {lower_bound} to {upper_bound}")
    
    # 5. Standardize text columns (if any)
    text_columns = df_cleaned.select_dtypes(include=['object']).columns
    for column in text_columns:
        # Convert to lowercase and strip whitespace
        if len(df_cleaned[column]) > 0:
            try:
                df_cleaned[column] = df_cleaned[column].str.lower().str.strip()
                print(f"Standardized text formatting in '{column}'")
            except:
                pass
    
    # Save the cleaned dataset
    if output_path is None:
        base_name = file_path.rsplit('.', 1)[0]
        ext = file_path.rsplit('.', 1)[1]
        output_path = f"{base_name}_cleaned.{ext}"
    
    if output_path.endswith('.csv'):
        df_cleaned.to_csv(output_path, index=False)
    elif output_path.endswith(('.xls', '.xlsx')):
        df_cleaned.to_excel(output_path, index=False)
    elif output_path.endswith('.json'):
        df_cleaned.to_json(output_path, orient='records')
    
    print(f"\nCleaned dataset shape: {df_cleaned.shape}")
    print(f"Saved cleaned dataset to: {output_path}")
    
    return df_cleaned

def generate_data_quality_report(df, output_path=None):
    """
    Generate a data quality report for the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    output_path : str, optional
        Path where the report will be saved
    """
    print("\n--- Generating Data Quality Report ---")
    
    report = {
        'data_types': df.dtypes,
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_values': df.nunique(),
        'descriptive_stats': df.describe(include='all').transpose()
    }
    
    # Display report
    print("\nData Types:")
    print(report['data_types'])
    
    print("\nMissing Values:")
    missing_df = pd.DataFrame({
        'Count': report['missing_values'],
        'Percentage': report['missing_percentage']
    })
    print(missing_df[missing_df['Count'] > 0])
    
    print("\nUnique Values:")
    print(report['unique_values'])
    
    print("\nSummary Statistics:")
    print(report['descriptive_stats'])
    
# Remove links
    if remove_links:
        # Match URLs starting with http, https, ftp, or www
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, replace_with, text)
        
        # Match URLs starting with www
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(www_pattern, replace_with, text)
    
    
    # Save the report if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write("# Data Quality Report\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Data Types\n")
            f.write(report['data_types'].to_string())
            
            f.write("\n\n## Missing Values\n")
            f.write(missing_df[missing_df['Count'] > 0].to_string())
            
            f.write("\n\n## Unique Values\n")
            f.write(report['unique_values'].to_string())
            
            f.write("\n\n## Summary Statistics\n")
            f.write(report['descriptive_stats'].to_string())
        
        print(f"Saved data quality report to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "fencing_data.csv"
    
    # Clean the dataset
    cleaned_df = clean_dataset(file_path)
    
    