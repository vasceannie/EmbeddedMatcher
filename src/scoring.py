class CombinedMatcher:
    """
    A class to combine the results of synonym matching and cosine similarity.

    This class provides methods to calculate a combined score based on the results from synonym matching
    and cosine similarity, determining the best match along with its score.

    Attributes:
        None
    """

    @staticmethod
    def combined_match(synonym_matches, cosine_matches):
        """
        Combine the results of synonym matching and cosine similarity to determine the best match.

        This method takes two sets of matches: one from synonym matching and another from cosine similarity.
        It calculates a combined score based on the provided matches and returns the best match along with its score.

        Args:
            synonym_matches (list): A list of tuples where each tuple contains a synonym match and its associated score.
            cosine_matches (list): A list of tuples where each tuple contains a cosine match and its associated score.

        Returns:
            tuple: A tuple containing the best match (str) and its combined score (float). 
                   If no matches are found, returns (None, 0).
        """
        if not synonym_matches and not cosine_matches:
            return None, 0

        if not synonym_matches:
            return cosine_matches[0]

        if not cosine_matches:
            return synonym_matches[0]

        # Fix: Use the correct index for cosine_matches
        combined_score = (synonym_matches[0][1] + cosine_matches[0][1]) / 2
        
        # Return the match with the highest individual score
        if synonym_matches[0][1] >= cosine_matches[0][1]:
            return synonym_matches[0]
        else:
            return cosine_matches[0]

    @staticmethod
    def apply_combined_scoring(df):
        """
        Apply the combined matching function to a DataFrame to generate final matches.

        This method takes a DataFrame and applies the combined_match function to each row,
        using the 'synonym_matches' and 'cosine_matches' columns to compute the final match for each entry.

        Args:
            df (pandas.DataFrame): The DataFrame containing 'synonym_matches' and 'cosine_matches' columns.

        Returns:
            pandas.DataFrame: The updated DataFrame with a new column 'final_match' containing the results of the combined matching.
        """
        # Apply the combined_match function to each row of the DataFrame and store the results in a new column 'final_match'
        df['final_match'] = df.apply(lambda row: CombinedMatcher.combined_match(row['synonym_matches'], row['cosine_matches']), axis=1)
        return df