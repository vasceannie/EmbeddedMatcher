import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import json
import numpy as np

# Load environment variables from the .env file to configure the application
load_dotenv()

# Add the parent directory to the system path to allow importing from the src module
# This is necessary for the test cases to access the EmbeddingManager class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
import tempfile
from src.utils import EmbeddingManager

class TestEmbeddingManager(unittest.TestCase):
    """
    Unit test class for the EmbeddingManager.

    This class contains test cases to verify the functionality of saving and loading
    BERT embeddings using different storage options such as CSV, PostgreSQL, and MongoDB.
    """

    def setUp(self):
        """
        Set up the test environment by creating a sample DataFrame.

        This method is called before each test case. It initializes a sample DataFrame
        with categories and random BERT embeddings for testing purposes.
        """
        # Create a sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'bert_embedding': [np.random.rand(768).tolist(), np.random.rand(768).tolist(), np.random.rand(768).tolist()]
        })
        self.connection_string = EmbeddingManager.get_postgres_connection_string()

    def test_save_load_csv(self):
        """
        Test saving and loading embeddings to and from a CSV file.

        This test verifies that embeddings can be saved to a CSV file and then loaded
        back correctly. It checks that the loaded DataFrame has the same shape and
        categories as the original DataFrame, and that the embeddings are close to the
        original values.
        """
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Save embeddings to CSV
            EmbeddingManager.save_embeddings(self.sample_df, temp_filename, db_type='csv')
            
            # Load embeddings from CSV
            loaded_df = EmbeddingManager.load_embeddings(temp_filename, db_type='csv')
            
            # Add a check to filter out null bytes
            loaded_df['bert_embedding'] = loaded_df['bert_embedding'].apply(
                lambda x: np.array(json.loads(x.replace('\x00', ''))) if isinstance(x, str) else x
            )
            
            # Check if the loaded DataFrame has the same shape and categories
            self.assertEqual(self.sample_df.shape, loaded_df.shape)
            pd.testing.assert_series_equal(self.sample_df['category'], loaded_df['category'])
            
            # Check if the loaded embeddings have the correct length
            self.assertEqual(len(loaded_df['bert_embedding'].iloc[0]), 768)

            # Check if the loaded embeddings are close to the original embeddings
            np.testing.assert_allclose(self.sample_df['bert_embedding'].iloc[0], loaded_df['bert_embedding'].iloc[0], rtol=1e-5)
        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    def test_save_load_postgres(self):
        """
        Test saving and loading embeddings to and from a PostgreSQL database.
        """
        engine = create_engine(self.connection_string)
        table_name = 'test_embeddings'

        # Save embeddings to PostgreSQL
        EmbeddingManager.save_embeddings(self.sample_df, self.connection_string, db_type='postgres')

        # Load embeddings from PostgreSQL
        loaded_df = EmbeddingManager.load_embeddings(self.connection_string, db_type='postgres')

        # Convert the 'bert_embedding' column back to lists
        loaded_df['bert_embedding'] = loaded_df['bert_embedding'].apply(
            lambda x: np.array(json.loads(x.replace('\x00', ''))) if isinstance(x, str) else x
        )

        # Check if the loaded DataFrame has the same shape and categories
        self.assertEqual(self.sample_df.shape, loaded_df.shape)
        pd.testing.assert_series_equal(self.sample_df['category'], loaded_df['category'])

        # Check if the loaded embeddings have the correct length
        self.assertEqual(len(loaded_df['bert_embedding'].iloc[0]), 768)

        # Check if the loaded embeddings are close to the original embeddings
        np.testing.assert_allclose(self.sample_df['bert_embedding'].iloc[0], loaded_df['bert_embedding'].iloc[0], rtol=1e-5)

    def test_unsupported_db_type(self):
        """
        Test handling of unsupported database types.

        This test verifies that attempting to save or load embeddings with an unsupported
        database type raises a ValueError.
        """
        with self.assertRaises(ValueError):
            EmbeddingManager.save_embeddings(self.sample_df, 'dummy', db_type='unsupported')
        
        with self.assertRaises(ValueError):
            EmbeddingManager.load_embeddings('dummy', db_type='unsupported')

if __name__ == '__main__':
    unittest.main()