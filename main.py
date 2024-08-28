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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_postgres_engine():
    connection_string = EmbeddingManager.get_postgres_connection_string()
    return create_engine(connection_string, connect_args={"connect_timeout": 10})

def save_checkpoint(df, table_name, engine):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.float64)):
            return int(obj) if isinstance(obj, np.int64) else float(obj)
        if isinstance(obj, tuple):
            return json.dumps(obj)
        return obj

    # Convert numpy.ndarray, list, and tuple columns to JSON strings
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].apply(lambda x: json.dumps(serialize(x)) if isinstance(x, (np.ndarray, list, tuple)) else x)
    
    df_copy.to_sql(table_name, engine, if_exists='replace', index=False)

def load_checkpoint(table_name):
    engine = get_postgres_engine()
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        logger.info(f"Loading checkpoint from table: {table_name}")
        df = pd.read_sql_table(table_name, engine)
        
        # Convert JSON strings back to lists or numpy arrays
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) and x.startswith('[') else x)
        
        return df
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
    engine = get_postgres_engine()

    # Step 1: Data Loading and Preprocessing
    checkpoint_table = 'checkpoint_preprocessed'
    source_df = load_checkpoint(checkpoint_table)
    target_df = load_target_data()
    
    if source_df is None:
        source_df = load_source_data()
        preprocessor = DataPreprocessor()
        source_df, target_df = preprocessor.preprocess_dataframes(source_df, target_df)
        save_checkpoint(source_df, checkpoint_table, engine)
    
    # Step 2: Synonym Matching
    checkpoint_table = 'checkpoint_synonym'
    synonym_df = load_checkpoint(checkpoint_table)
    
    if synonym_df is None:
        synonym_matcher = SynonymMatcher()
        synonym_df = synonym_matcher.apply_synonym_matching(source_df, target_df)
        save_checkpoint(synonym_df, checkpoint_table, engine)
    else:
        source_df = synonym_df

    # Step 3: Cosine Similarity Matching
    checkpoint_table = 'checkpoint_cosine'
    cosine_df = load_checkpoint(checkpoint_table)
    
    if cosine_df is None:
        cosine_matcher = CosineSimilarityMatcher()
        # Generate embeddings for both source and target DataFrames
        source_df = cosine_matcher.generate_embeddings(source_df)
        target_df = cosine_matcher.generate_embeddings(target_df)
        cosine_df = cosine_matcher.apply_cosine_matching(source_df, target_df)
        save_checkpoint(cosine_df, checkpoint_table, engine)
    else:
        source_df = cosine_df

    # Step 4: Combined Scoring
    checkpoint_table = 'checkpoint_combined'
    combined_df = load_checkpoint(checkpoint_table)
    
    if combined_df is None:
        combined_matcher = CombinedMatcher()
        combined_df = combined_matcher.apply_combined_scoring(source_df)
        save_checkpoint(combined_df, checkpoint_table, engine)
    else:
        source_df = combined_df

    # Step 5: Output
    if 'final_match' in source_df.columns:
        for _, row in source_df.iterrows():
            final_match = row['final_match']
            if isinstance(final_match, tuple) and len(final_match) == 2:
                print(f"Source: {row.get('classification_name', 'N/A')}, Best Match: {final_match[0]}, Score: {final_match[1]:.2f}")
            else:
                print(f"Source: {row.get('classification_name', 'N/A')}, Final Match: {final_match}")
    else:
        print("Final matches not available. Please ensure the combined scoring step was completed.")

    # Optional: Save embeddings to PostgreSQL
    if 'bert_embedding' in source_df.columns and 'bert_embedding' in target_df.columns:
        embedding_manager = EmbeddingManager()
        source_id_col = 'classification_name' if 'classification_name' in source_df.columns else source_df.columns[0]
        target_id_col = 'category' if 'category' in target_df.columns else target_df.columns[0]
        embedding_manager.save_embeddings(source_df[[source_id_col, 'bert_embedding']], engine, db_type='postgres')
        embedding_manager.save_embeddings(target_df[[target_id_col, 'bert_embedding']], engine, db_type='postgres')
    else:
        print("BERT embeddings not available. Skipping embedding save step.")

    # Step 6: Validation
    if 'final_match' in source_df.columns and 'classification_name' in source_df.columns:
        true_labels = source_df['classification_name']
        predicted_labels = source_df['final_match'].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) == 2 else x)
        metrics = ValidationMetrics.calculate_metrics(true_labels, predicted_labels)
        ValidationMetrics.print_metrics(metrics)
    else:
        print("Final matches or classification names not available. Skipping validation step.")

    # Save final results
    source_df.to_csv('matched_results.csv', index=False)

if __name__ == "__main__":
    main()
