# %% [markdown]
# This code is used to install specific Python libraries (packages) in your environment using pip, which is the Python package manager. These libraries are commonly used for data manipulation, visualization, and scientific computing. 

# %%
%pip install pandas 
%pip install numpy
%pip nltk 

# %% [markdown]
# Load the Dataset

# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('C:\\Users\\Admin\Desktop\\simba ai\\imdb_top_250 - imdb_top_250.csv')
df
#with pd.option_context('display.max_rows', None, 'display.max_columns',None):

   # display(df)

# %% [markdown]
# General information about the Dataset

# %%
df.info()


# %% [markdown]
# Generate summary statistics of a DataFrame. It provides a quick overview of the central tendency, spread, and shape of the distribution of your dataset's numerical columns.

# %%
df.describe()

# %% [markdown]
# Used to check for missing or null values (often represented as NaN—Not a Number) in a DataFrame.

# %%
# check the missing columns
df.isnull().sum()

# %% [markdown]
# This line of code is used to check for duplicate rows in the DataFrame df and count the total number of duplicated rows.

# %%
# to check duplicates
df.duplicated().sum()


# %% [markdown]
# This code is used to remove duplicate rows from the DataFrame df.

# %%
# remove duplicate columns
df.drop_duplicates(inplace=True)

# %% [markdown]
# This line of code is used to get the unique values in the vote_count column of the DataFrame df and store them in the variable store_val.

# %%
store_val=df.vote_count.unique()
store_val


# %% [markdown]
# This line of code is used to sort the rows of the DataFrame df based on the values in the vote_count column, in ascending order.

# %%
# use of lamda function
# Apply the lambda function to filter the DataFrame
#df = pd.DataFrame(data)
# sorted_df = df.sort_values(by='array', key=lambda col: col.str.lower())
# print(sorted_df)
df.sort_values("vote_count", ascending=True)

# %% [markdown]
# This line of code is used to count the total number of non-null entries across all columns in the DataFrame df_cleaned.

# %%
df_cleaned.count().sum()

# %% [markdown]
# The code you provided performs several tasks on a DataFrame containing movie data, specifically focusing on converting the vote_count column to numeric values and handling some possible inconsistencies in the data. 

# %%
import pandas as pd

# Load data
data = pd.read_csv(r'data\movies.csv')  # Replace with the correct path to your CSV file

# Create DataFrame (assuming 'Values' is the column you're working with)
df = pd.DataFrame(data)

# Function to convert values to numeric
def convert_to_numeric(value):
    if 'M' in value: 
        return float(value.strip('()M')) * 1
    elif 'K' in value: 
        return float(value.strip('()K')) * 0.001
    else:
        return float(value.strip('()'))

# Apply the function to convert values
df['vote_count'] = df['vote_count'].apply(convert_to_numeric)

# Optionally, if you want to make all values equal, you could set them to the most common value, for example:
df['vote_count'] = df['vote_count'].mode()[0]

# Show the results
print(df)


# %% [markdown]
# This line of code is used to check and count duplicate rows in the DataFrame df.

# %%
# to check duplicates
df.duplicated().sum()


# %% [markdown]
# Removes duplicate rows in the DataFrame df based on the title column.
# The resulting DataFrame df_cleaned contains only unique titles.

# %%
# to check duplicates
df_cleaned = df.drop_duplicates(subset=['title'])


df_cleaned['title'].tolist()


# %%
df_cleaned.duplicated().sum()

# %%
df_cleaned.count().sum()

# %% [markdown]
# This line of code is used to save the cleaned DataFrame df_cleaned into a CSV file called 'movies_cleaned.csv'.

# %%
df_cleaned.to_csv('movies_cleaned.csv', index=False)


