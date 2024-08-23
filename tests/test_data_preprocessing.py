import sys
import os
import pandas as pd
import unittest
from unittest.mock import patch, Mock
from io import StringIO

# Insert the parent directory into the system path to allow importing modules from the src directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()

    def test_preprocess(self):
        text = "This is a Sample Text with Punctuation!"
        expected = "sample text punctuation"
        self.assertEqual(self.preprocessor.preprocess(text), expected)

    @patch('pandas.read_csv')
    def test_load_data_non_hierarchical(self, mock_read_csv):
        # Mock CSV content for non-hierarchical data
        source_data = pd.DataFrame({
            'id': [1, 2],
            'classification_name': ['Electronics', 'Clothing']
        })
        target_data = pd.DataFrame({
            'id': [1, 2],
            'category_name': ['Gadgets', 'Apparel']
        })
        
        mock_read_csv.side_effect = [source_data, target_data]
        
        source_df, target_df = self.preprocessor.load_data('source.csv', 'target.csv')
        
        self.assertIn('classification_name', source_df.columns)
        self.assertIn('category_name', target_df.columns)
        self.assertEqual(len(source_df), 2)
        self.assertEqual(len(target_df), 2)

    @patch('pandas.read_csv')
    def test_load_data_hierarchical(self, mock_read_csv):
        # Mock CSV content for hierarchical data
        source_data = pd.DataFrame({
            'id': [1, 2],
            'classification_name': ['Electronics', 'Clothing']
        })
        target_data = pd.DataFrame({
            'categoryID': [1, 2],
            'Lv1_category_name': ['Customs', 'Facilities'],
            'Lv2_category_name': ['Custom Tax or Duty', 'Building & Facility Maintenance & Repair Services'],
            'Lv3_category_name': ['Custom Tax or Duty', 'Building Maintenance & Repair Services'],
            'Lv4_category_name': ['Custom Tax or Duty', 'Building Construction Services']
        })
        
        mock_read_csv.side_effect = [source_data, target_data]
        
        source_df, target_df = self.preprocessor.load_data('source.csv', 'target.csv')
        
        self.assertIn('classification_name', source_df.columns)
        self.assertIn('category_name', target_df.columns)
        self.assertEqual(len(source_df), 2)
        self.assertEqual(len(target_df), 2)
        self.assertEqual(target_df['category_name'].iloc[0], 'Customs > Custom Tax or Duty > Custom Tax or Duty > Custom Tax or Duty')
        self.assertEqual(target_df['category_name'].iloc[1], 'Facilities > Building & Facility Maintenance & Repair Services > Building Maintenance & Repair Services > Building Construction Services')

    @patch('pandas.read_csv')
    def test_load_data_invalid_target(self, mock_read_csv):
        # Mock CSV content for invalid target data
        source_csv = StringIO("id,classification_name\n1,Electronics\n2,Clothing")
        target_csv = StringIO("id,invalid_column\n1,Gadgets\n2,Apparel")
        
        mock_read_csv.side_effect = [pd.read_csv(source_csv), pd.read_csv(target_csv)]
        
        with self.assertRaises(ValueError):
            self.preprocessor.load_data('source.csv', 'target.csv')
    @patch('pandas.read_csv')
    def test_load_data_missing_category(self, mock_read_csv):
        # Mock CSV content for missing category column
        source_csv = StringIO("id,classification_name\n1,Electronics\n2,Clothing")
        target_csv = StringIO("id,missing_category\n1,Gadgets\n2,Apparel")
        
        mock_read_csv.side_effect = [pd.read_csv(source_csv), pd.read_csv(target_csv)]
        
        with self.assertRaises(ValueError) as context:
            self.preprocessor.load_data('source.csv', 'target.csv')
        
        actual_message = str(context.exception)
        expected_message = "Source data must have a 'classification_name' column."
        self.assertEqual(actual_message, expected_message)

    def test_preprocess_dataframes(self):
        source_df = pd.DataFrame({'category': ['Electronics', 'Clothing']})
        target_df = pd.DataFrame({'category': ['Gadgets', 'Apparel']})
        
        processed_source, processed_target = self.preprocessor.preprocess_dataframes(source_df, target_df)
        
        self.assertIn('processed_category', processed_source.columns)
        self.assertIn('processed_category', processed_target.columns)
        self.assertEqual(processed_source['processed_category'].iloc[0], 'electronics')
        self.assertEqual(processed_target['processed_category'].iloc[1], 'apparel')

if __name__ == '__main__':
    unittest.main()