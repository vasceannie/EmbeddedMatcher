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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_postgres_engine():
    """
    Create and return a SQLAlchemy engine for PostgreSQL connection.

    This function retrieves the PostgreSQL connection string using the EmbeddingManager class
    and creates a SQLAlchemy engine with a connection timeout of 10 seconds.

    Returns:
        sqlalchemy.engine.Engine: The SQLAlchemy engine for PostgreSQL connection.
    """
    connection_string = EmbeddingManager.get_postgres_connection_string()
    return create_engine(connection_string, connect_args={"connect_timeout": 10})

def serialize(obj):
    """
    Serialize various data types to JSON-compatible formats.

    This function handles the serialization of numpy arrays, pandas DataFrames, Series, tuples,
    and other data types to ensure they can be converted to JSON format.

    Args:
        obj: The object to serialize.

    Returns:
        The serialized object in a JSON-compatible format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    if isinstance(obj, (np.int64, np.float64)):
        return int(obj) if isinstance(obj, np.int64) else float(obj)  # Convert numpy int/float to Python int/float
    if isinstance(obj, tuple):
        return {'__tuple__': True, 'items': list(obj)}  # Convert tuple to dictionary with tuple flag
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return {'__pandas__': True, 'data': obj.to_dict()}  # Convert pandas DataFrame/Series to dictionary with pandas flag
    return str(obj)  # Convert other types to string

def deserialize(obj):
    """
    Deserialize JSON-compatible formats back to their original data types.

    This function handles the deserialization of JSON-compatible formats back to numpy arrays,
    pandas DataFrames, Series, tuples, and other data types.

    Args:
        obj: The JSON-compatible object to deserialize.

    Returns:
        The deserialized object in its original data type.
    """
    if isinstance(obj, dict):
        if '__tuple__' in obj:
            return tuple(deserialize(item) for item in obj['items'])  # Convert dictionary with tuple flag back to tuple
        if '__pandas__' in obj:
            data = obj['data']
            if isinstance(data, dict):
                if all(isinstance(v, dict) for v in data.values()):
                    return pd.DataFrame({k: deserialize(v) for k, v in data.items()})  # Convert dictionary with pandas flag back to DataFrame
                else:
                    return pd.Series(deserialize(data))  # Convert dictionary with pandas flag back to Series
    elif isinstance(obj, list):
        # Check if the list should be converted back to a numpy array
        if all(isinstance(item, (int, float, list)) for item in obj):
            try:
                return np.array(obj)  # Convert list back to numpy array
            except ValueError:
                return [deserialize(item) for item in obj]  # Recursively deserialize list items
        return [deserialize(item) for item in obj]  # Recursively deserialize list items
    return obj  # Return the object as is if no conversion is needed

def save_checkpoint(df, table_name, engine):
    """
    Save a DataFrame to a PostgreSQL table as a checkpoint.

    This function saves a copy of the DataFrame to a PostgreSQL table. If an error occurs,
    it saves the DataFrame to a CSV file as a fallback.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        table_name (str): The name of the PostgreSQL table.
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine for PostgreSQL connection.
    """
    df_copy = df.copy()
    
    # Convert object columns to strings
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' or df_copy[col].dtypes.name == 'object':
            df_copy[col] = df_copy[col].apply(lambda x: str(x) if x is not None else None)
    
    try:
        df_copy.to_sql(table_name, engine, if_exists='replace', index=False)  # Save DataFrame to PostgreSQL table
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        csv_filename = f"{table_name}_checkpoint.csv"
        df_copy.to_csv(csv_filename, index=False)  # Save DataFrame to CSV file as fallback
        logger.info(f"Checkpoint saved to CSV: {csv_filename}")

def load_checkpoint(table_name):
    """
    Load a checkpoint DataFrame from a PostgreSQL table.

    This function loads a DataFrame from a PostgreSQL table if it exists. It also handles
    the deserialization of JSON strings back to their original data types.

    Args:
        table_name (str): The name of the PostgreSQL table.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame or None if the table does not exist.
    """
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
    """
    Load the source data from a CSV file.

    This function loads the source data from the 'distinct_source_class.csv' file.

    Returns:
        pandas.DataFrame or None: The loaded source DataFrame or None if an error occurs.
    """
    try:
        source_df = pd.read_csv('distinct_source_class.csv')
        logger.info("source_df loaded from CSV.")
        return source_df
    except Exception as e:
        logger.error(f"Error loading source_df: {e}")
        return None

def load_target_data():
    """
    Load the target data from a CSV file.

    This function loads the target data from the 'target_taxonomy.csv' file.

    Returns:
        pandas.DataFrame or None: The loaded target DataFrame or None if an error occurs.
    """
    try:
        target_df = pd.read_csv('target_taxonomy.csv')
        logger.info("target_df loaded from CSV.")
        return target_df
    except Exception as e:
        logger.error(f"Error loading target_df: {e}")
        return None

def process_chunk(chunk, target_df):
    """
    Process a chunk of the source DataFrame by applying synonym matching, cosine similarity matching, and combined scoring.

    This function initializes the matchers within the worker process, applies synonym matching,
    generates embeddings, applies cosine similarity matching, and calculates combined scores.

    Args:
        chunk (pandas.DataFrame): The chunk of the source DataFrame to process.
        target_df (pandas.DataFrame): The target DataFrame.

    Returns:
        pandas.DataFrame: The processed chunk with matches and scores.
    """
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

    This function performs the following steps:
    1. Data Loading and Preprocessing
    2. Synonym Matching
    3. Cosine Similarity Matching
    4. Combined Scoring
    5. Saving Results
    6. Validation
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
    num_processes = mp.cpu_count() - 6

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
        connection_string = str(engine.url)
        embedding_manager.save_embeddings(
            final_df[[source_id_col, 'bert_embedding']], 
            connection_string,  # Use the connection string instead of the engine
            db_type='postgres'
        )
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
