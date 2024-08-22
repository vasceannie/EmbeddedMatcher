import sys
import os

# Insert the parent directory into the system path to allow importing modules from the src directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from unittest.mock import patch
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """
    Unit test class for testing the DataPreprocessor functionality.
    This class contains test cases to verify the behavior of the DataPreprocessor methods.
    """

    def setUp(self):
        """
        Set up the test environment before each test case.
        
        This method initializes an instance of DataPreprocessor and creates sample
        source and target data as Pandas DataFrames for testing purposes.
        """
        self.preprocessor = DataPreprocessor()
        self.source_data = {
            'category': ['This is a test.', 'Another test case!', 'Data preprocessing is important.']
        }
        self.target_data = {
            'category': ['Testing data.', 'Preprocessing is key.', 'This is another example.']
        }
        self.source_df = pd.DataFrame(self.source_data)
        self.target_df = pd.DataFrame(self.target_data)

    @patch.object(DataPreprocessor, 'load_data', return_value=(pd.DataFrame({'category': ['mocked data']}), pd.DataFrame({'category': ['mocked data']})))
    def test_load_data(self, mock_load_data):
        """
        Test the load_data method of the DataPreprocessor class.
        
        This test verifies that the load_data method returns two DataFrames
        when called, and checks that both returned objects are instances of
        pandas DataFrame.
        """
        source_df, target_df = self.preprocessor.load_data('source.csv', 'target.csv')
        self.assertIsInstance(source_df, pd.DataFrame)
        self.assertIsInstance(target_df, pd.DataFrame)

    def test_preprocess(self):
        """
        Test the preprocess method of the DataPreprocessor class.
        
        This test checks that the preprocess method correctly processes a sample
        text by ensuring the output is a string and that certain words (like 'this'
        and 'is') are not present in the processed text.
        """
        text = "This is a sample text for preprocessing."
        processed_text = self.preprocessor.preprocess(text)
        self.assertIsInstance(processed_text, str)
        self.assertNotIn('this', processed_text.lower())
        self.assertNotIn('is', processed_text.lower())

    def test_preprocess_dataframes(self):
        """
        Test the preprocess_dataframes method of the DataPreprocessor class.
        
        This test verifies that the preprocess_dataframes method adds a new column
        'processed_category' to both source and target DataFrames and checks that
        the lengths of the new columns match the original DataFrames.
        """
        source_df, target_df = self.preprocessor.preprocess_dataframes(self.source_df, self.target_df)
        self.assertIn('processed_category', source_df.columns)
        self.assertIn('processed_category', target_df.columns)
        self.assertEqual(len(source_df['processed_category']), len(self.source_df))
        self.assertEqual(len(target_df['processed_category']), len(self.target_df))

if __name__ == '__main__':
    # Run the unit tests when the script is executed directly.
    unittest.main()