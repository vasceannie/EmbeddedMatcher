import joblib
import pandas as pd

def save_embeddings(df, filename):
    """
    Save the BERT embeddings and their associated categories to a CSV file.

    This function takes a DataFrame containing categories and their corresponding BERT embeddings,
    and saves the relevant data to a specified CSV file. The resulting CSV will have two columns:
    'category' and 'bert_embedding'.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'category' and 'bert_embedding' columns.
        filename (str): The name of the file where the DataFrame will be saved.
    """
    # Select the relevant columns and save the DataFrame to a CSV file
    df[['category', 'bert_embedding']].to_csv(filename, index=False)

def load_embeddings(filename):
    """
    Load BERT embeddings from a CSV file into a DataFrame.

    This function reads a CSV file containing categories and their corresponding BERT embeddings,
    and loads the data into a DataFrame. The 'bert_embedding' column is processed to convert
    the string representations of embeddings back into their original format using joblib.

    Args:
        filename (str): The name of the file from which to load the embeddings.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded categories and their BERT embeddings.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    # Convert the string representations of embeddings back into numpy arrays
    df['bert_embedding'] = df['bert_embedding'].apply(lambda x: joblib.loads(x))
    return df