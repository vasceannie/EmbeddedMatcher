import logging
import pandas as pd

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
        print(f"Debug - synonym_matches: {synonym_matches}")
        print(f"Debug - cosine_matches: {cosine_matches}")

        if not synonym_matches and not cosine_matches:
            return (None, 0)

        if not synonym_matches:
            return max(cosine_matches, key=lambda x: x[1])

        if not cosine_matches:
            return max(synonym_matches, key=lambda x: x[1])

        # Return the match with the highest individual score
        best_synonym_match = max(synonym_matches, key=lambda x: x[1])
        best_cosine_match = max(cosine_matches, key=lambda x: x[1])

        result = best_synonym_match if best_synonym_match[1] >= best_cosine_match[1] else best_cosine_match
        print(f"Debug - combined_match result: {result}")
        return result

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
        def combined_match_wrapper(row):
            result = self.combined_match(
                row.get('synonym_matches', []),
                row.get('cosine_matches', [])
            )
            print(f"Debug - combined_match result: {result}")
            print(f"Debug - result type: {type(result)}")
            
            if isinstance(result, pd.DataFrame):
                print(f"Debug - DataFrame columns: {result.columns}")
                print(f"Debug - DataFrame shape: {result.shape}")
                # If it's a DataFrame, take the first row as a tuple
                return tuple(result.iloc[0])
            elif isinstance(result, tuple):
                return result
            else:
                print(f"Debug - Unexpected result type: {type(result)}")
                return (None, 0)  # Default value if result is unexpected

        # Apply the function row by row and collect the results
        results = []
        for _, row in df.iterrows():
            result = combined_match_wrapper(row)
            results.append(result)
            print(f"Debug - Appended result: {result}")

        # Assign the results to the 'final_match' column
        df['final_match'] = results
        return df