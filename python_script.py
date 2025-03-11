#Step 1: Import Necessary Libraries
#You need to install and import the necessary libraries:
pip install pandas numpy

# Then, import them in your Python script:
import pandas as pd
import numpy as np
#Step 2: Load Your Dataset
# Load dataset
df = pd.read_csv("your_dataset.csv")

# Display the first few rows to understand the structure
print(df.head())

#Step 3: Handle Missing Values
# Handling missing data is one of the first tasks in data cleaning. You can either drop rows or fill missing values.
#3.1 Drop Missing Values
#You can drop rows with missing values using dropna():
# Drop rows with any missing values
df = df.dropna()

# Drop rows where specific columns have missing values
df = df.dropna(subset=['column_name'])
#3.2 Fill Missing Values
#You can fill missing values with a default value, mean, median, or mode using fillna():
# Fill missing values with a constant value
df['column_name'] = df['column_name'].fillna(0)

# Fill with the mean of the column
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# Fill with the mode of the column
df['column_name'] = df['column_name'].fillna(df['column_name'].mode()[0])

#Step 4: Remove Duplicates
#You may want to remove duplicate rows in the dataset.
# Remove duplicate rows
df = df.drop_duplicates()

#Step 5: Convert Data Types
#Sometimes, the data type of a column is not what you expect. You can change it using astype().
# Convert a column to a numeric type
df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')  # 'coerce' converts invalid parsing to NaN

# Convert a column to a datetime type
df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

#Step 6: Standardize Text Data
#Ensure that text data is consistent. For example, strip extra spaces, convert to lowercase, or fix typos.
# Strip leading/trailing spaces and convert to lowercase
df['column_name'] = df['column_name'].str.strip().str.lower()
df['column_name'] = df['column_name'].str.strip().str.lower()

#Step 7: Handle Outliers (Optional)
#Outliers can sometimes affect the accuracy of your analysis. You can either remove or correct them.
# Removing outliers by applying a threshold
df = df[df['column_name'] < 100]  # Remove rows where 'column_name' has values greater than 100

#Step 8: Feature Engineering (Optional)
#This step involves creating new columns based on existing data or transforming features to improve the dataset for modeling.
# Example: Creating a new column 'age' based on 'dob' (date of birth)
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365

#Step 9: Save the Cleaned Data
#Once you've cleaned the dataset, you can save it back to a new file:
# Save the cleaned dataset to a new CSV file
df.to_csv("cleaned_dataset.csv", index=False)   









