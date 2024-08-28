import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import string

class SynonymMatcher:
    """
    A class to perform synonym matching between source and target categories
    using WordNet to find synonyms and calculate similarity based on shared
    synonyms.
    """

    def __init__(self, threshold=0.1):
        """
        Initialize the SynonymMatcher with a similarity threshold.

        Args:
            threshold (float): The minimum similarity score required to consider
                               a match. Default is 0.1.
        """
        self.threshold = threshold
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text):
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords
        tokens = [token for token in tokens if token not in string.punctuation and token not in self.stop_words]
        return tokens

    def get_synonyms(self, word):
        """
        Retrieve synonyms for a given word using WordNet.

        Args:
            word (str): The input word for which to find synonyms.

        Returns:
            set: A set of synonyms for the input word.
        """
        synonyms = set([word])  # Include the original word
        # Iterate through the synsets of the given word
        for syn in wn.synsets(word):
            # Add all lemma names from the synset to the synonyms set
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return synonyms  # Return the set of synonyms

    def synonym_match(self, source_category, target_df):
        matches = []
        source_tokens = set(self.tokenize(source_category))
        source_synonyms = set()
        for token in source_tokens:
            source_synonyms.update(self.get_synonyms(token))
        
        for _, row in target_df.iterrows():
            target_category = row.get('category_name') or row.get('processed_category') or row.iloc[0]
            target_tokens = set(self.tokenize(str(target_category)))
            target_synonyms = set()
            for token in target_tokens:
                target_synonyms.update(self.get_synonyms(token))
            
            # Calculate Jaccard similarity
            intersection = len(source_synonyms.intersection(target_synonyms))
            union = len(source_synonyms.union(target_synonyms))
            match_score = intersection / union if union > 0 else 0
            
            # Only append matches with a score greater than the threshold
            matches.append((target_category, match_score))
        
        # Sort matches by score in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match for match in matches if match[1] > self.threshold]

    def apply_synonym_matching(self, source_df, target_df):
        """
        Apply synonym matching to the source DataFrame.

        Args:
            source_df (pandas.DataFrame): The DataFrame containing source categories.
            target_df (pandas.DataFrame): The DataFrame containing target categories.

        Returns:
            pandas.DataFrame: The updated source DataFrame with a new column
                              containing the matches and their similarity scores.
        """
        result_df = source_df.copy()  # Create a copy of the source DataFrame to avoid modifying the original
        
        # Ensure 'processed_category' exists in result_df
        if 'processed_category' not in result_df.columns:
            raise KeyError("'processed_category' column is missing in result_df")
        
        # Apply the synonym_match method to each processed category in the source DataFrame
        result_df['synonym_matches'] = result_df['processed_category'].apply(
            lambda x: self.synonym_match(x, target_df)  # Use a lambda function to apply synonym matching
        )
        print(f"Applied synonym matching: {result_df}")  # Debug print to show the result DataFrame
        return result_df  # Return the DataFrame with matches