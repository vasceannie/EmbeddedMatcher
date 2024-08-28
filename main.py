import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError
from src.data_preprocessing import DataPreprocessor
from src.synonym_matching import SynonymMatcher
from src.cosine_similarity import CosineSimilarityMatcher
from src.scoring import CombinedMatcher
from src.utils import EmbeddingManager
from src.validation_metrics import ValidationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_postgres_engine():
    connection_string = EmbeddingManager.get_postgres_connection_string()
    return create_engine(connection_string, connect_args={"connect_timeout": 10})

def save_checkpoint(df, table_name, engine):
    # Convert numpy.ndarray columns to lists
    for col in df.columns:
        if df[col].dtype == np.object:
            if isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def load_checkpoint(table_name):
    engine = get_postgres_engine()
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        logger.info(f"Loading checkpoint from table: {table_name}")
        return pd.read_sql_table(table_name, engine)
    logger.info(f"No checkpoint found for table: {table_name}")
    return None

def load_source_data():
    try:
        source_df = pd.read_csv('distinct_source_class.csv')
        logger.info("source_df loaded from CSV.")
        return source_df
    except Exception as e:
        logger.error(f"Error loading source_df: {e}")
        return None

def load_target_data():
    try:
        target_df = pd.read_csv('target_taxonomy.csv')
        logger.info("target_df loaded from CSV.")
        return target_df
    except Exception as e:
        logger.error(f"Error loading target_df: {e}")
        return None

def main():
    """
    Main function to execute the category matching and scoring process.
    """
    # Step 1: Data Loading and Preprocessing
    checkpoint_table = 'checkpoint_preprocessed'
    engine = get_postgres_engine()
    source_df = load_checkpoint(checkpoint_table)
    target_df = load_target_data()  # Ensure target_df is loaded before preprocessing
    if source_df is None:
        logger.info("source_df is None after loading checkpoint. Attempting to load from CSV.")
        source_df = load_source_data()
        if source_df is None:
            raise ValueError("source_df is None. Please check the data loading process.")
        else:
            logger.info("source_df loaded successfully from CSV.")
            # Preprocess the data
            preprocessor = DataPreprocessor()
            source_df, target_df = preprocessor.preprocess_dataframes(source_df, target_df)
            save_checkpoint(source_df, checkpoint_table, engine)
    else:
        logger.info("source_df loaded successfully from checkpoint.")
        # Ensure 'processed_category' column exists
        if 'processed_category' not in source_df.columns:
            preprocessor = DataPreprocessor()
            source_df, target_df = preprocessor.preprocess_dataframes(source_df, target_df)
            save_checkpoint(source_df, checkpoint_table, engine)
    
    if target_df is None:
        raise ValueError("target_df is None. Please check the data loading process.")
    
    logger.info(f"Source DataFrame columns after preprocessing: {source_df.columns.tolist()}")
    logger.info(f"Target DataFrame columns after preprocessing: {target_df.columns.tolist()}")

    # Step 2: Synonym Matching
    checkpoint_table = 'checkpoint_synonym'
    synonym_matcher = SynonymMatcher()
    source_df = load_checkpoint(checkpoint_table)
    if source_df is None:
        source_df = load_source_data()  # Ensure source_df is loaded before synonym matching
        if source_df is None:
            logger.error("source_df is None before applying synonym matching.")
            return
        # Ensure preprocessing is applied if loading from CSV
        preprocessor = DataPreprocessor()
        source_df, target_df = preprocessor.preprocess_dataframes(source_df, target_df)
        source_df = synonym_matcher.apply_synonym_matching(source_df, target_df)
        if source_df is None:
            raise ValueError("source_df is None after synonym matching. Please check the synonym matching process.")
        save_checkpoint(source_df, checkpoint_table, engine)
    else:
        logger.info("source_df loaded successfully from checkpoint_synonym.")

    # Step 3: Cosine Similarity Matching
    checkpoint_table = 'checkpoint_cosine'
    source_df = load_checkpoint(checkpoint_table)
    if source_df is None or source_df.empty:
        logger.info("No valid checkpoint found for table: checkpoint_cosine")
        source_df = load_checkpoint('checkpoint_synonym')
        if source_df is None or source_df.empty:
            logger.error("source_df is None or empty after loading from synonym checkpoint.")
            return
        cosine_matcher = CosineSimilarityMatcher()
        # Ensure target_df has 'processed_category' column
        logger.info(f"Target DataFrame columns before generating embeddings: {target_df.columns}")
        if 'processed_category' not in target_df.columns:
            # Preprocess target_df to add 'processed_category' column
            preprocessor = DataPreprocessor()
            _, target_df = preprocessor.preprocess_dataframes(source_df, target_df)
            if 'processed_category' not in target_df.columns:
                raise KeyError("The column 'processed_category' does not exist in the target DataFrame before generating embeddings.")
        source_df = cosine_matcher.generate_embeddings(source_df)
        target_df = cosine_matcher.generate_embeddings(target_df)
        source_df = cosine_matcher.apply_cosine_matching(source_df, target_df)
        save_checkpoint(source_df, checkpoint_table, engine)
    else:
        logger.info("source_df loaded successfully from checkpoint_cosine.")

    # Add debugging information
    logger.info(f"source_df shape after cosine matching: {source_df.shape}")
    logger.info(f"source_df columns after cosine matching: {source_df.columns}")
    logger.info(f"First few rows of source_df after cosine matching:\n{source_df.head()}")

    # Step 4: Combined Scoring
    checkpoint_table = 'checkpoint_combined'
    source_df = load_checkpoint(checkpoint_table)
    if source_df is None or source_df.empty:
        logger.info("No valid checkpoint found for table: checkpoint_combined")
        combined_matcher = CombinedMatcher()
        # We don't need to load from checkpoint_cosine again, as we already have source_df
        if source_df is None or source_df.empty:
            logger.error("source_df is None or empty before combined scoring")
            return
        try:
            source_df = combined_matcher.apply_combined_scoring(source_df)
            save_checkpoint(source_df, checkpoint_table, engine)
        except ValueError as e:
            logger.error(f"Error in combined scoring: {e}")
            # Add more debugging information
            logger.error(f"source_df shape: {source_df.shape}")
            logger.error(f"source_df columns: {source_df.columns}")
            logger.error(f"First few rows of source_df:\n{source_df.head()}")
            return
    else:
        logger.info("source_df loaded successfully from checkpoint_combined.")

    # Add more debugging information
    logger.info(f"source_df shape after combined scoring: {source_df.shape}")
    logger.info(f"source_df columns after combined scoring: {source_df.columns}")
    logger.info(f"First few rows of source_df after combined scoring:\n{source_df.head()}")

    # Step 5: Output
    logger.info("Matching Results:")
    for _, row in source_df.iterrows():
        logger.info(f"Source: {row['classification_name']}, Best Match: {row['final_match'][0]}, Score: {row['final_match'][1]:.2f}")

    # Optional: Save embeddings to PostgreSQL
    embedding_manager = EmbeddingManager()
    embedding_manager.save_embeddings(source_df[['classification_name', 'bert_embedding']], engine, db_type='postgres')
    embedding_manager.save_embeddings(target_df[['category', 'bert_embedding']], engine, db_type='postgres')

    # Step 6: Validation
    true_labels = source_df['classification_name']
    predicted_labels = source_df['final_match'].apply(lambda x: x[0])
    metrics = ValidationMetrics.calculate_metrics(true_labels, predicted_labels)
    ValidationMetrics.print_metrics(metrics)

    # Save final results
    source_df.to_csv('matched_results.csv', index=False)

if __name__ == "__main__":
    main()
