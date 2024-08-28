import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from pymongo import MongoClient
import numpy as np
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EmbeddingManager:
    """
    A class to manage the saving and loading of BERT embeddings.

    This class provides methods to save BERT embeddings and their associated categories to various storage options,
    including PostgreSQL and MongoDB.
    """

    @staticmethod
    def get_postgres_connection_string():
        """
        Construct the PostgreSQL connection string from environment variables.
        """
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'your_database_name')
        user = os.getenv('POSTGRES_USER', 'your_username')
        password = os.getenv('POSTGRES_PASSWORD', 'your_password')
        return f"postgresql://{user}:{password}@{host}:{port}/{db}?connect_timeout=60"

    @staticmethod
    def save_embeddings(df, path_or_connection_string, db_type='postgres'):
        """
        Save the BERT embeddings and their associated categories to the specified database.

        Args:
            df (pandas.DataFrame): The DataFrame containing 'category' and 'bert_embedding' columns.
            path_or_connection_string (str): The path to save to CSV or the connection string for PostgreSQL or MongoDB.
            db_type (str): The type of database ('csv', 'postgres', or 'mongo'). Default is 'csv'.

        Returns:
            None
        """
        if db_type == 'csv':
            df.to_csv(path_or_connection_string, index=False)
        elif db_type == 'postgres':
            engine = create_engine(path_or_connection_string)
            df.to_sql('embeddings', engine, if_exists='replace', index=False)
        elif db_type == 'mongo':
            client = MongoClient(path_or_connection_string)
            db = client['embedding_db']
            collection = db['embeddings']  # {{ edit_1 }} Define the collection
            collection.insert_many(df.to_dict('records'))
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    @staticmethod
    def load_embeddings(path_or_connection_string, db_type='postgres'):
        """
        Load BERT embeddings from the specified database into a DataFrame.

        Args:
            path_or_connection_string (str): The path to load from CSV or the connection string for PostgreSQL or MongoDB.
            db_type (str): The type of database ('csv', 'postgres', or 'mongo'). Default is 'csv'.

        Returns:
            pandas.DataFrame: A DataFrame containing the loaded categories and their BERT embeddings.
        """
        if db_type == 'csv':
            df = pd.read_csv(path_or_connection_string)
            # Convert the 'bert_embedding' column back to numpy arrays
            df['bert_embedding'] = df['bert_embedding'].apply(
                lambda x: np.array(json.loads(x.replace('\x00', ''))) if isinstance(x, str) and x.startswith('[') else np.array(x, dtype=float)
            )
            return df
        elif db_type == 'postgres':
            engine = create_engine(path_or_connection_string)
            df = pd.read_sql_table('embeddings', engine)
            # Convert the 'bert_embedding' column back to numpy arrays
            df['bert_embedding'] = df['bert_embedding'].apply(
                lambda x: np.array(json.loads(x.replace('{', '[').replace('}', ']'))) 
                if isinstance(x, str) and (x.startswith('[') or x.startswith('{')) 
                else np.array(x, dtype=float)
            )
            return df
        elif db_type == 'mongo':
            client = MongoClient(path_or_connection_string)
            db = client['embedding_db']
            collection = db['embeddings']  # {{ edit_1 }} Define the collection
            # Load BERT embeddings from the specified database into a DataFrame.
            df = pd.DataFrame(list(collection.find()))
            # Sanitize the 'bert_embedding' column to remove any entries with null bytes
            df = df[df['bert_embedding'].apply(lambda x: isinstance(x, str) and '\x00' not in x)]
            # Convert the 'bert_embedding' column back to numpy arrays
            df['bert_embedding'] = df['bert_embedding'].apply(
                lambda x: np.array(json.loads(x.replace('{', '[').replace('}', ']'))) 
                if isinstance(x, str) and (x.startswith('[') or x.startswith('{')) 
                else np.array(x, dtype=float)
            )
            return df
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")