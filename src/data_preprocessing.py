import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Load the English NLP model from spaCy
nlp = spacy.load('en_core_web_sm')
# Initialize the WordNet lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()
# Create a set of English stopwords to filter out common words
stop_words = set(stopwords.words('english'))

def load_data(source_path, target_path):
    """
    Load data from CSV files into Pandas DataFrames.

    Args:
        source_path (str): The file path to the source CSV file.
        target_path (str): The file path to the target CSV file.

    Returns:
        tuple: A tuple containing two DataFrames, the source DataFrame and the target DataFrame.
    """
    # Read the source and target CSV files into DataFrames
    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)
    return source_df, target_df

def preprocess(text):
    """
    Preprocess the input text by normalizing it.

    This function converts the text to lowercase, removes punctuation,
    tokenizes the text, and removes stopwords. It also lemmatizes the
    remaining words to their base forms.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text as a single string.
    """
    # Convert text to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    return ' '.join(tokens)

def preprocess_dataframes(source_df, target_df):
    """
    Preprocess the 'category' column in both source and target DataFrames.

    This function applies the preprocessing steps defined in the
    preprocess function to the 'category' column of both DataFrames,
    creating a new column 'processed_category' in each.

    Args:
        source_df (pandas.DataFrame): The DataFrame containing source categories.
        target_df (pandas.DataFrame): The DataFrame containing target categories.

    Returns:
        tuple: A tuple containing the updated source and target DataFrames.
    """
    # Apply the preprocess function to the 'category' column of both DataFrames
    source_df['processed_category'] = source_df['category'].apply(preprocess)
    target_df['processed_category'] = target_df['category'].apply(preprocess)
    return source_df, target_df