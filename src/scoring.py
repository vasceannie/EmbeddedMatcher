import logging
import pandas as pd

class CombinedMatcher:
    """
    A class to combine the results of synonym matching and cosine similarity.

    This class provides methods to calculate a combined score based on the results from synonym matching
    and cosine similarity, determining the best match along with its score.

    Attributes:
        synonym_weight (float): The weight assigned to synonym matching scores.
        cosine_weight (float): The weight assigned to cosine similarity scores.
    """
    def __init__(self, synonym_weight=0.3, cosine_weight=0.7):
        """
        Initializes the CombinedMatcher with specified weights for synonym and cosine matching.

        Args:
            synonym_weight (float): The weight assigned to synonym matching scores.
            cosine_weight (float): The weight assigned to cosine similarity scores.
        """
        self.synonym_weight = synonym_weight
        self.cosine_weight = cosine_weight

    def combined_match(self, synonym_matches, cosine_matches):
        """
        Combine the results of synonym matching and cosine similarity using weights.

        Args:
            synonym_matches (list): A list of tuples where each tuple contains a synonym match and its associated score.
            cosine_matches (list): A list of tuples where each tuple contains a cosine match and its associated score.

        Returns:
            tuple: A tuple containing the best match (str) and its combined score (float). 
                   If no matches are found, returns (None, 0).
        """
        print(f"Debug - synonym_matches: {synonym_matches}")
        print(f"Debug - cosine_matches: {cosine_matches}")

        # Filter out non-tuple elements from synonym_matches
        synonym_matches = [match for match in synonym_matches if isinstance(match, tuple)]
        # Filter out non-tuple elements from cosine_matches
        cosine_matches = [match for match in cosine_matches if isinstance(match, tuple)]

        # If both lists are empty, return (None, 0)
        if not synonym_matches and not cosine_matches:
            return (None, 0)

        # Calculate weighted scores for each synonym match
        weighted_matches = []
        for match, score in synonym_matches:
            weighted_score = score * self.synonym_weight
            weighted_matches.append((match, weighted_score, 'synonym'))
        
        # Calculate weighted scores for each cosine match
        for match, score in cosine_matches:
            weighted_score = score * self.cosine_weight
            weighted_matches.append((match, weighted_score, 'cosine'))

        # Find the match with the highest weighted score
        if weighted_matches:
            best_match = max(weighted_matches, key=lambda x: x[1])
            return (best_match[0], best_match[1])
        else:
            return (None, 0)

    @staticmethod
    def apply_combined_scoring(df):
        """
        Apply combined scoring to a DataFrame by combining synonym and cosine matches.

        This method processes each row of the DataFrame, applying the combined_match method to determine
        the best match and its score, and stores the result in a new 'final_match' column.

        Args:
            df (pandas.DataFrame): The DataFrame containing 'classification_name', 'processed_category',
                                   'synonym_matches', and 'cosine_matches' columns.

        Returns:
            pandas.DataFrame: The DataFrame with an additional 'final_match' column containing the best matches
                              and their combined scores.

        Raises:
            ValueError: If the 'final_match' column does not contain both the match and the score.
        """
        if df is None:
            logging.error("DataFrame is None in apply_combined_scoring")
            return None

        # Apply the combined_match method to each row and store the result in 'final_match' column
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
        """
        Apply combined scoring to a DataFrame by combining synonym and cosine matches.

        This method processes each row of the DataFrame, applying the combined_match method to determine
        the best match and its score, and stores the result in a new 'final_match' column.

        Args:
            df (pandas.DataFrame): The DataFrame containing 'classification_name', 'processed_category',
                                   'synonym_matches', and 'cosine_matches' columns.

        Returns:
            pandas.DataFrame: The DataFrame with an additional 'final_match' column containing the best matches
                              and their combined scores.
        """
    def apply_combined_scoring(self, df):
        """
        Apply combined scoring to the DataFrame.

        This method applies the combined_match function to each row of the DataFrame,
        combining synonym matches and cosine similarity matches.

        Args:
            df (pandas.DataFrame): The input DataFrame containing synonym and cosine matches.

        Returns:
            pandas.DataFrame: The input DataFrame with additional columns for the best match and its score.
        """
        def combined_match_wrapper(row):
            """
            Wrapper function to apply combined_match to a DataFrame row.

            This function extracts the synonym and cosine matches from the row, applies the combined_match method,
            and handles the result appropriately.

            Args:
                row (pandas.Series): A row from the DataFrame.

            Returns:
                tuple: The best match and its combined score.
            """
            result = self.combined_match(
                row.get('synonym_matches', []),
                row.get('cosine_matches', [])
            )
            print(f"Debug - combined_match result: {result}")
            print(f"Debug - result type: {type(result)}")
            
            # Handle the result based on its type
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

        # Apply the combined_match_wrapper function row by row and collect the results
        results = []
        for _, row in df.iterrows():
            result = combined_match_wrapper(row)
            results.append(result)
            print(f"Debug - Appended result: {result}")

        # Assign the results to the 'final_match' column
        df['final_match'] = results
        return df