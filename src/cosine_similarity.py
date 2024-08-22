from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the BERT tokenizer and model for embedding generation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """
    Generate BERT embeddings for a given text.

    Args:
        text (str): The input text for which to generate embeddings.

    Returns:
        numpy.ndarray: The mean of the last hidden states from the BERT model,
                       representing the embedding of the input text.
    """
    # Tokenize the input text and prepare it for the BERT model
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return the mean of the last hidden states as the embedding
    return outputs.last_hidden_state.mean(dim=1).numpy()

def generate_embeddings(df):
    """
    Generate BERT embeddings for the 'processed_category' column in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'processed_category' column.

    Returns:
        pandas.DataFrame: The DataFrame with an additional 'bert_embedding' column
                          containing the generated embeddings.
    """
    # Apply the get_bert_embedding function to each processed category
    df['bert_embedding'] = df['processed_category'].apply(get_bert_embedding)
    return df

def cosine_match(source_embedding, target_embeddings):
    """
    Calculate the cosine similarity between a source embedding and a list of target embeddings.

    Args:
        source_embedding (numpy.ndarray): The embedding of the source item.
        target_embeddings (list of numpy.ndarray): A list of target embeddings to compare against.

    Returns:
        tuple: A tuple containing the best matching target embedding and its similarity score.
    """
    # Compute cosine similarities between the source embedding and all target embeddings
    similarities = cosine_similarity(source_embedding.reshape(1, -1), target_embeddings)
    
    # Find the index of the best matching target embedding
    best_match_idx = similarities.argmax()
    
    # Return the best matching target embedding and its similarity score
    return target_embeddings[best_match_idx], similarities.max()

def apply_cosine_matching(source_df, target_df):
    """
    Apply cosine matching to find the best matches for each source embedding in the target embeddings.

    Args:
        source_df (pandas.DataFrame): The DataFrame containing source embeddings.
        target_df (pandas.DataFrame): The DataFrame containing target embeddings.

    Returns:
        pandas.DataFrame: The source DataFrame with an additional 'cosine_matches' column
                          containing the best matches and their similarity scores.
    """
    # Apply cosine_match to each source embedding and store the results in a new column
    source_df['cosine_matches'] = source_df['bert_embedding'].apply(lambda x: cosine_match(x, target_df['bert_embedding'].tolist()))
    return source_df