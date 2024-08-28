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
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_postgres_engine():
    connection_string = EmbeddingManager.get_postgres_connection_string()
    return create_engine(connection_string, connect_args={"connect_timeout": 10})

def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Ensure ndarray is converted to list
    if isinstance(obj, (np.int64, np.float64)):
        return int(obj) if isinstance(obj, np.int64) else float(obj)
    if isinstance(obj, tuple):
        return {'__tuple__': True, 'items': list(obj)}
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return {'__pandas__': True, 'data': obj.to_dict()}
    return str(obj)

def deserialize(obj):
    if isinstance(obj, dict):
        if '__tuple__' in obj:
            return tuple(deserialize(item) for item in obj['items'])
        if '__pandas__' in obj:
            data = obj['data']
            if isinstance(data, dict):
                if all(isinstance(v, dict) for v in data.values()):
                    return pd.DataFrame({k: deserialize(v) for k, v in data.items()})
                else:
                    return pd.Series(deserialize(data))
    elif isinstance(obj, list):
        # Check if the list should be converted back to a numpy array
        if all(isinstance(item, (int, float, list)) for item in obj):
            try:
                return np.array(obj)
            except ValueError:
                return [deserialize(item) for item in obj]
        return [deserialize(item) for item in obj]
    return obj

def save_checkpoint(df, table_name, engine):
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].apply(lambda x: json.dumps(serialize(x)) if x is not None else None)  # Handle None values
    
    try:
        df_copy.to_sql(table_name, engine, if_exists='replace', index=False)
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        # Fallback: save to CSV
        csv_filename = f"{table_name}_checkpoint.csv"
        df_copy.to_csv(csv_filename, index=False)
        logger.info(f"Checkpoint saved to CSV: {csv_filename}")

def load_checkpoint(table_name):
    engine = get_postgres_engine()
    inspector = inspect(engine)
    if inspector.has_table(table_name):
        logger.info(f"Loading checkpoint from table: {table_name}")
        df = pd.read_sql_table(table_name, engine)
        
        # Convert JSON strings back to lists, numpy arrays, tuples, or pandas objects
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: deserialize(json.loads(x)) if isinstance(x, str) and (x.startswith('[') or x.startswith('{')) else x)
        
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

def process_chunk(chunk, target_df):
    # Initialize matchers within the worker process
    synonym_matcher = SynonymMatcher()
    cosine_matcher = CosineSimilarityMatcher()
    combined_matcher = CombinedMatcher()
    
    # Synonym Matching
    chunk = synonym_matcher.apply_synonym_matching(chunk, target_df)
    
    # Cosine Similarity Matching
    chunk = cosine_matcher.generate_embeddings(chunk)
    chunk = cosine_matcher.apply_cosine_matching(chunk, target_df)
    
    # Combined Scoring
    chunk = combined_matcher.apply_combined_scoring(chunk)
    
    return chunk

def main():
    """
    Main function to execute the category matching and scoring process.
    """
    engine = get_postgres_engine()

    # Step 1: Data Loading and Preprocessing
    checkpoint_table = 'checkpoint_preprocessed'
    source_df = load_checkpoint(checkpoint_table)
    target_df = load_target_data()
    
    if source_df is None or 'processed_category' not in source_df.columns or 'processed_category' not in target_df.columns:
        source_df = load_source_data()
        target_df = load_target_data()
        preprocessor = DataPreprocessor()
        source_df, target_df = preprocessor.preprocess_dataframes(source_df, target_df)
        save_checkpoint(source_df, checkpoint_table, engine)
    
    # Ensure 'processed_category' exists in target_df
    if 'processed_category' not in target_df.columns:
        logger.warning("'processed_category' not found in target_df. Using 'category' if available.")
        if 'category' in target_df.columns:
            target_df['processed_category'] = target_df['category']
        else:
            raise KeyError("Neither 'processed_category' nor 'category' found in target_df.")

    # Generate embeddings for target DataFrame
    cosine_matcher = CosineSimilarityMatcher()
    target_df = cosine_matcher.generate_embeddings(target_df)

    # Determine the number of processes to use (leave one core free for system tasks)
    num_processes = mp.cpu_count() - 1

    # Split the source DataFrame into chunks
    chunks = np.array_split(source_df, num_processes)

    # Create a pool of worker processes
    pool = mp.Pool(processes=num_processes)

    # Process chunks in parallel
    try:
        results = pool.starmap(process_chunk, [(chunk, target_df) for chunk in chunks])
    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        pool.close()
        pool.join()
        raise

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Combine the results
    final_df = pd.concat(results, ignore_index=True)

    # Save the final results
    save_checkpoint(final_df, 'final_results', engine)

    # Output results
    if 'final_match' in final_df.columns:
        for _, row in final_df.iterrows():
            final_match = row['final_match']
            if isinstance(final_match, tuple) and len(final_match) == 2:
                print(f"Source: {row.get('classification_name', 'N/A')}, Best Match: {final_match[0]}, Score: {final_match[1]:.2f}")
            else:
                print(f"Source: {row.get('classification_name', 'N/A')}, Final Match: {final_match}")
    else:
        print("Final matches not available. Please ensure the combined scoring step was completed.")

    # Optional: Save embeddings to PostgreSQL
    if 'bert_embedding' in final_df.columns and 'bert_embedding' in target_df.columns:
        embedding_manager = EmbeddingManager()
        source_id_col = 'classification_name' if 'classification_name' in final_df.columns else final_df.columns[0]
        target_id_col = 'category' if 'category' in target_df.columns else target_df.columns[0]
        embedding_manager.save_embeddings(final_df[[source_id_col, 'bert_embedding']], engine, db_type='postgres')
        embedding_manager.save_embeddings(target_df[[target_id_col, 'bert_embedding']], engine, db_type='postgres')
    else:
        print("BERT embeddings not available. Skipping embedding save step.")

    # Validation
    if 'final_match' in final_df.columns and 'classification_name' in final_df.columns:
        true_labels = final_df['classification_name']
        predicted_labels = final_df['final_match'].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) == 2 else x)
        metrics = ValidationMetrics.calculate_metrics(true_labels, predicted_labels)
        ValidationMetrics.print_metrics(metrics)
    else:
        print("Final matches or classification names not available. Skipping validation step.")

    # Save final results to CSV
    final_df.to_csv('matched_results.csv', index=False)

if __name__ == "__main__":
    main()
