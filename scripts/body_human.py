{
 cells,
  
 []
   "cell_type": "code",
   "execution_count": none,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
    # Correct in Python 
   " execution_count = None" : 
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv('human_body_data.csv')\n",
    "df.head()"
   ]
  [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "data wrangling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# Check the shape\n",
    "print(\"Our dataset has {} samples and {} features.\".format(*df.shape))\n",
    "\n",
    "# Check for missing values\n",
    "missing = df.isnull().sum()\n",
    "missing = missing[missing > 0]\n",
    "missing_percentage = missing / df.shape[0] * 100\n",
    "missing_info = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percentage})\n",
    "print(missing_info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# Imputing summary and content features\n",
    "# Since it is categorical, we will replace missing values with the mode\n",
    "df['summary'] = df['summary'].fillna(df['summary'].mode()[0])\n",
    "df['content'] = df['summary'].fillna(df['content'].mode()[0])\n",
    "\n",
    "# Drop the links and url features as they are not useful\n",
    "#df = df.drop(['url', 'links'], axis=1)\n",
    "\n",
    "# Get categorical summary statistics\n",
    "cat_summary = df.describe(include=['object']).T\n",
    "\n",
    "# Add additional information\n",
    "cat_summary['missing'] = df.isnull().sum()\n",
    "cat_summary['unique'] = df.nunique()\n",
    "cat_summary['dtype'] = df.dtypes\n",
    "\n",
    "# Display the results\n",
    "print(\"\\nCategorical Variables Summary:\")\n",
    "print(cat_summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# We want to do away completely with \\n, \\r, \\t, and other special characters across the dataset\n",
    df = df.replace(r'\\n', ' ', regex=True)\n",
    "df = df.replace(r'\\r', ' ', regex=True)\n",
    "df = df.replace(r'\\t', ' ', regex=True)\n",
    "df = df.replace(r'\\\\n', ' ', regex=True)\n",
    "df = df.replace(r'\\\\r', ' ', regex=True)\n",
    "df = df.replace(r'\\\\t', ' ', regex=True)\n",
    "df = df.replace(r'\\s+', ' ', regex=True)\n",
    "\n",
    "# Check the first 5 rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# Drop url and links columns\n",
    "df = df.drop(['url', 'links'], axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# Define appropriate content for human body\n",
    "cfi_summary = \"Human anatomy and physiology are treated in many different articles. For detailed discussions of specific tissues, organs, and systems, see human blood; cardiovascular system; human digestive system; human endocrine system; renal system; skin; human muscle system; nervous system; human reproductive system; human respiration; human sensory reception; and human skeletal system. For a description of how the body develops, from conception through old age, see aging; growth; prenatal development; and human development.\"\n",
    "cfi_content = \"Humans are, of course, animals—more particularly, members of the order Primates in the subphylum Vertebrata of the phylum Chordata. Like all chordates, the human animal has a bilaterally symmetrical body that is characterized at some point during its development by a dorsal supporting rod (the notochord), gill slits in the region of the pharynx, and a hollow dorsal nerve cord. Of these features, the first two are present only during the embryonic stage in the human; the notochord is replaced by the vertebral column, and the pharyngeal gill slits are lost completely. The dorsal nerve cord is the spinal cord in humans; it remains throughout life.\"\n",
    "# Update the specific row\n",
    "df.loc[2, 'summary'] = cfi_summary\n",
    "df.loc[2, 'content'] = cfi_content\n",
    "\n",
    "# Verify update\n",
    "print(df.iloc[2])\n",
    "\n",
    "# Check for missing values\n",
    "print(\"Missing values in the dataset: \", df.isnull().sum().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   }, 
   "outputs": [],
   "source": [
    "# Save the cleaned dataset\n",
    "df.to_csv('cleaned_human_body_data.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
      
