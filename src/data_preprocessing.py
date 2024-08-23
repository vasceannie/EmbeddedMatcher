import pandas as pd
import spacy
from spacy.cli import download
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

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
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        # Join the tokens back into a single string
        return ' '.join(tokens)

    def preprocess_dataframes(self, source_df, target_df):
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
        # Apply the preprocess function to the 'category' column of the source DataFrame
        source_df['processed_category'] = source_df['category'].apply(self.preprocess)
        
        # Apply the preprocess function to the 'category' column of the target DataFrame
        target_df['processed_category'] = target_df['category'].apply(self.preprocess)
        
        return source_df, target_df