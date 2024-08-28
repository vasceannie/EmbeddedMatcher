import logging

class CombinedMatcher:
    """
    A class to combine the results of synonym matching and cosine similarity.

    This class provides methods to calculate a combined score based on the results from synonym matching
    and cosine similarity, determining the best match along with its score.
    """
    def __init__(self, synonym_weight=0.3, cosine_weight=0.7):
        self.synonym_weight = synonym_weight
        self.cosine_weight = cosine_weight

    @staticmethod
    def combined_match(synonym_matches, cosine_matches):
        """
        Combine the results of synonym matching and cosine similarity to determine the best match.

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
            return max(cosine_matches, key=lambda x: x[1])

        if not cosine_matches:
            return max(synonym_matches, key=lambda x: x[1])

        # Return the match with the highest individual score
        best_synonym_match = max(synonym_matches, key=lambda x: x[1])
        best_cosine_match = max(cosine_matches, key=lambda x: x[1])

        return best_synonym_match if best_synonym_match[1] >= best_cosine_match[1] else best_cosine_match

    @staticmethod
    def apply_combined_scoring(df):
        if df is None:
            logging.error("DataFrame is None in apply_combined_scoring")
            return None

        df['final_match'] = df.apply(lambda row: CombinedMatcher.debug_combined_match(
            row['classification_name'],
            row['processed_category'],
            row['synonym_matches'],
            row['cosine_matches']
        ), axis=1)

        # Ensure that final_match contains two values (match and score)
        if df['final_match'].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0).min() < 2:
            raise ValueError("final_match must contain both the match and the score")

        return df

    def apply_combined_scoring(self, df):
        df['final_match'] = df.apply(lambda row: self.combined_match(
            row.get('synonym_matches', []),
            row.get('cosine_matches', [])
        ), axis=1)
        return df