def combined_match(synonym_matches, cosine_matches):
    """
    Combine the results of synonym matching and cosine similarity to determine the best match.

    This function takes two sets of matches: one from synonym matching and another from cosine similarity.
    It calculates a combined score based on the provided matches and returns the best match along with its score.

    Args:
        synonym_matches (list): A list of tuples where each tuple contains a synonym match and its associated score.
        cosine_matches (list): A list of tuples where each tuple contains a cosine match and its associated score.

    Returns:
        tuple: A tuple containing the best match (str) and its combined score (float). 
               If no matches are found, returns (None, 0).
    """
    # Check if both synonym matches and cosine matches are available
    if synonym_matches and cosine_matches:
        # Calculate the combined score as the average of the best synonym match score and the best cosine match score
        combined_score = (synonym_matches[0][1] + cosine_matches[1]) / 2
        # Return the best synonym match and the combined score
        return synonym_matches[0][0], combined_score
    elif synonym_matches:
        # If only synonym matches are available, return the best synonym match and its score
        return synonym_matches[0][0], synonym_matches[0][1]
    elif cosine_matches:
        # If only cosine matches are available, return the best cosine match and its score
        return cosine_matches[0], cosine_matches[1]
    else:
        # If no matches are found, return None and a score of 0
        return None, 0

def apply_combined_scoring(df):
    """
    Apply the combined matching function to a DataFrame to generate final matches.

    This function takes a DataFrame and applies the combined_match function to each row,
    using the 'synonym_matches' and 'cosine_matches' columns to compute the final match for each entry.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'synonym_matches' and 'cosine_matches' columns.

    Returns:
        pandas.DataFrame: The updated DataFrame with a new column 'final_match' containing the results of the combined matching.
    """
    # Apply the combined_match function to each row of the DataFrame and store the results in a new column 'final_match'
    df['final_match'] = df.apply(lambda row: combined_match(row['synonym_matches'], row['cosine_matches']), axis=1)
    return df