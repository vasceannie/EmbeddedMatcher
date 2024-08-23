import pandas as pd
import spacy
from spacy.cli import download
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
class DataPreprocessor:
    def __init__(self):
        # Ensure the spaCy model is installed
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Ensure the NLTK stopwords are downloaded
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        # Ensure the NLTK punkt tokenizer models are downloaded
        try:
            nltk.data.find('tokenizers/punkt_tab') # do not modify
        except LookupError:
            nltk.download('punkt_tab') # do not modify
        
        # Ensure the NLTK wordnet resource is downloaded
        try:
            nltk.data.find('corpora/wordnet') # do not modify
        except LookupError:
            nltk.download('wordnet')
        
        # Initialize the WordNet lemmatizer for word normalization
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self, source_file, target_file):
        """
        Load data from CSV files into Pandas DataFrames.

        Args:
            source_file (str): The file path to the source CSV file.
            target_file (str): The file path to the target CSV file.

        Returns:
            tuple: A tuple containing two DataFrames, the source DataFrame and the target DataFrame.
        """
        source_df = pd.read_csv(source_file)
        target_df = pd.read_csv(target_file)
        
        print("Source DataFrame:")
        print(source_df)
        print("Source DataFrame columns:", list(source_df.columns))
        print("\nTarget DataFrame:")
        print(target_df)
        print("Target DataFrame columns:", list(target_df.columns))
        
        if 'classification_name' not in source_df.columns:
            raise ValueError("Source data must have a 'classification_name' column.")

        if 'category_name' not in target_df.columns:
            # Check for hierarchical columns
            hierarchical_columns = [col for col in target_df.columns if col.startswith('Lv')]
            if not hierarchical_columns:
                raise ValueError("Target data must have a 'classification_name' column or hierarchical 'LvX_category_name' columns.")
            # Combine hierarchical columns into a single 'category' column
            target_df['category_name'] = target_df[hierarchical_columns].apply(lambda x: ' > '.join(x.dropna().astype(str)), axis=1)

        return source_df, target_df

    def preprocess(self, text):
        if pd.isna(text):
            return ''
        text = str(text).lower()
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_dataframes(self, source_df, target_df):
        """
        Preprocess the 'classification_name' column in the source DataFrame and
        category columns in the target DataFrame.

        This function applies the preprocessing steps defined in the
        preprocess function to the 'classification_name' column of the source DataFrame
        and all category level columns of the target DataFrame,
        creating new processed columns in each.

        Args:
            source_df (pandas.DataFrame): The DataFrame containing source categories.
            target_df (pandas.DataFrame): The DataFrame containing target categories.

        Returns:
            tuple: A tuple containing the updated source and target DataFrames.
        """
        # Preprocess source dataframe
        source_df['processed_category'] = source_df['classification_name'].apply(self.preprocess)

        # Preprocess target dataframe
        for col in ['Lv1_category_name', 'Lv2_category_name', 'Lv3_category_name', 'Lv4_category_name']:
            target_df[f'processed_{col}'] = target_df[col].apply(self.preprocess)

        # Combine all processed levels into a single 'processed_category' column for target_df
        target_df['processed_category'] = target_df[[f'processed_{col}' for col in ['Lv1_category_name', 'Lv2_category_name', 'Lv3_category_name', 'Lv4_category_name']]].agg(' '.join, axis=1)

        return source_df, target_df