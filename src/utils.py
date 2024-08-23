import pandas as pd
import sqlalchemy
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
    including CSV files, PostgreSQL, MongoDB, and universal database types.
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
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    @staticmethod
    def save_embeddings(df, destination, db_type='csv'):
        """
        Save the BERT embeddings and their associated categories to the specified destination.

        Args:
            df (pandas.DataFrame): The DataFrame containing 'category' and 'bert_embedding' columns.
            destination (str): The destination where the DataFrame will be saved (filename for CSV, connection string for databases).
            db_type (str): The type of database ('csv', 'postgres', 'mongo', or 'universal'). Default is 'csv'.
        """
        df = df.copy()
        df['bert_embedding'] = df['bert_embedding'].apply(lambda x: x.tolist())

        load_dotenv()  # Load environment variables from .env file
        if db_type == 'csv':
            df['bert_embedding'] = df['bert_embedding'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            df.to_csv(destination, index=False)
        elif db_type == 'postgres':
            connection_string = EmbeddingManager.get_postgres_connection_string()
            engine = sqlalchemy.create_engine(connection_string)
            df['bert_embedding'] = df['bert_embedding'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            df.to_sql('embeddings', engine, if_exists='replace', index=False)
        elif db_type == 'mongo':
            client = MongoClient(destination)
            db = client['embeddings_db']
            collection = db['embeddings']
            records = df.to_dict(orient='records')
            collection.insert_many(records)
        elif db_type == 'universal':
            # Implement universal database saving logic here
            pass
        else:
            raise ValueError("Unsupported database type. Use 'csv', 'postgres', 'mongo', or 'universal'.")

    @staticmethod
    def load_embeddings(destination, db_type='csv'):
        """
        Load BERT embeddings from the specified source into a DataFrame.

        Args:
            destination (str): The source from which to load the embeddings (filename for CSV, connection string for databases).
            db_type (str): The type of database ('csv', 'postgres', 'mongo', or 'universal'). Default is 'csv'.

        Returns:
            pandas.DataFrame: A DataFrame containing the loaded categories and their BERT embeddings.
        """
        load_dotenv()  # Load environment variables from .env file
        if db_type == 'csv':
            df = pd.read_csv(destination)
            df['bert_embedding'] = df['bert_embedding'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
        elif db_type == 'postgres':
            connection_string = EmbeddingManager.get_postgres_connection_string()
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql('embeddings', engine)
            df['bert_embedding'] = df['bert_embedding'].apply(lambda x: np.array(json.loads(x)))
        elif db_type == 'mongo':
            client = MongoClient(destination)
            db = client['embeddings_db']
            collection = db['embeddings']
            records = collection.find()
            df = pd.DataFrame(list(records))
        elif db_type == 'universal':
            # Implement universal database loading logic here
            pass
        else:
            raise ValueError("Unsupported database type. Use 'csv', 'postgres', 'mongo', or 'universal'.")

        return df