from nltk.corpus import wordnet as wn

class SynonymMatcher:
    """
    A class to perform synonym matching between source and target categories
    using WordNet to find synonyms and calculate similarity based on shared
    synonyms.
    """

    def __init__(self, threshold=0.3):
        """
        Initialize the SynonymMatcher with a similarity threshold.

        Args:
            threshold (float): The minimum similarity score required to consider
                               a match. Default is 0.3.
        """
        self.threshold = threshold

    def get_synonyms(self, word):
        """
        Retrieve synonyms for a given word using WordNet.

        Args:
            word (str): The input word for which to find synonyms.

        Returns:
            set: A set of synonyms for the input word.
        """
        synonyms = set()  # Initialize an empty set to store synonyms
        # Iterate through the synsets of the given word
        for syn in wn.synsets(word):
            # Add all lemma names from the synset to the synonyms set
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
            # Add lemma names from hypernyms (more general terms)
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    synonyms.add(lemma.name())
            # Add lemma names from hyponyms (more specific terms)
            for hyponym in syn.hyponyms():
                for lemma in hyponym.lemmas():
                    synonyms.add(lemma.name())
            # Add lemma names from similar terms
            for related_form in syn.similar_tos():
                for lemma in related_form.lemmas():
                    synonyms.add(lemma.name())
        return synonyms  # Return the set of synonyms

    def synonym_match(self, source_category, target_df):
        """
        Match the source category against target categories based on synonyms.

        Args:
            source_category (str): The source category to match.
            target_df (pandas.DataFrame): A DataFrame containing target
                                           categories with a 'processed_category' column.

        Returns:
            list: A list of tuples containing matching target categories and their
                   similarity scores.
        """
        source_words = source_category.split()  # Split the source category into words
        source_synonyms = set()  # Initialize a set to hold synonyms for the source category
        
        # Collect synonyms for each token in the source category
        for word in source_words:
            source_synonyms.update(self.get_synonyms(word))  # Update the set with synonyms of each word
        
        matches = []  # Initialize a list to hold matches
        
        # Iterate through target categories to find matches
        for index, row in target_df.iterrows():
            target_words = row['processed_category'].split()  # Split the target category into words
            target_synonyms = set()  # Initialize a set to hold synonyms for the target category
            
            # Collect synonyms for each token in the target category
            for word in target_words:
                target_synonyms.update(self.get_synonyms(word))  # Update the set with synonyms of each word
            
            # Calculate similarity based on shared synonyms
            common_synonyms = source_synonyms.intersection(target_synonyms)  # Find common synonyms
            if common_synonyms:  # If there are common synonyms
                match_score = len(common_synonyms) / len(source_synonyms)  # Calculate match score
                if match_score >= self.threshold:  # Check if the score meets the threshold
                    matches.append((row['Lv4_category_name'], match_score))  # Append the match and score to the list
        
        print(f"Matches for {source_category}: {matches}")  # Debug print to show matches found
        return sorted(matches, key=lambda x: x[1], reverse=True)[:5]  # Return top 5 matches

    def apply_synonym_matching(self, source_df, target_df):
        """
        Apply synonym matching to the source DataFrame.

        Args:
            source_df (pandas.DataFrame): The DataFrame containing source categories.
            target_df (pandas.DataFrame): The DataFrame containing target categories.

        Returns:
            pandas.DataFrame: The source DataFrame with an additional 'synonym_matches' column
                              containing the matches and their similarity scores.
        """
        result_df = source_df.copy()  # Create a copy of the source DataFrame to avoid modifying the original
        # Apply the synonym_match method to each processed category in the source DataFrame
        result_df['synonym_matches'] = result_df['processed_category'].apply(
            lambda x: self.synonym_match(x, target_df)  # Use a lambda function to apply synonym matching
        )
        print(f"Applied synonym matching: {result_df}")  # Debug print to show the result DataFrame
        return result_df  # Return the DataFrame with matches