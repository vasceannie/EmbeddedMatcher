import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarityMatcher:
    """
    A class to compute BERT embeddings and perform cosine similarity matching.

    This class utilizes the BERT model to generate embeddings for text data and
    calculates cosine similarity between these embeddings to find the best matches.

    Attributes:
        tokenizer (BertTokenizer): The BERT tokenizer for processing text.
        model (BertModel): The BERT model for generating embeddings.
    """

    def __init__(self):
        """
        Initializes the CosineSimilarityMatcher by loading the BERT tokenizer and model.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embedding(self, text):
        """
        Generate BERT embeddings for a given text.

        Args:
            text (str): The input text for which to generate embeddings.

        Returns:
            numpy.ndarray: The mean of the last hidden states from the BERT model,
                           representing the embedding of the input text.
        """
        # Tokenize the input text and prepare it for the BERT model
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        
        # Disable gradient calculation for efficiency
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Return the mean of the last hidden states as the embedding
        return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

    def generate_embeddings(self, df):
        """
        Generate BERT embeddings for the 'processed_category' column in a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing a 'processed_category' column.

        Returns:
            pandas.DataFrame: The DataFrame with an additional 'bert_embedding' column
                              containing the generated embeddings.
        """
        # Apply the get_bert_embedding function to each processed category
        df['bert_embedding'] = df['processed_category'].apply(self.get_bert_embedding)
        return df

    def cosine_match(self, source_embedding, target_embeddings):
        """
        Calculate the cosine similarity between a source embedding and a list of target embeddings.

        Args:
            source_embedding (numpy.ndarray): The embedding of the source item.
            target_embeddings (list of numpy.ndarray): A list of target embeddings to compare against.

        Returns:
            tuple: A tuple containing the best matching target embedding and its similarity score.
        """
        # Ensure source_embedding is 2D
        source_embedding = source_embedding.reshape(1, -1)
        # Convert target_embeddings to a 2D numpy array
        target_embeddings = np.array(target_embeddings)
        if target_embeddings.ndim == 1:
            target_embeddings = target_embeddings.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(source_embedding, target_embeddings)
        
        # Find the index of the best matching target embedding
        best_match_idx = similarities.argmax()
        
        # Return the best matching target embedding and its similarity score
        return target_embeddings[best_match_idx], similarities.max()

    def apply_cosine_matching(self, source_df, target_df):
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
        source_df['cosine_matches'] = source_df['bert_embedding'].apply(
            lambda x: self.cosine_match(x, target_df['bert_embedding'].tolist())
        )
        return source_df