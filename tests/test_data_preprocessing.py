import unittest
import pandas as pd
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.source_data = {
            'category': ['This is a test.', 'Another test case!', 'Data preprocessing is important.']
        }
        self.target_data = {
            'category': ['Testing data.', 'Preprocessing is key.', 'This is another example.']
        }
        self.source_df = pd.DataFrame(self.source_data)
        self.target_df = pd.DataFrame(self.target_data)

    def test_load_data(self):
        source_df, target_df = self.preprocessor.load_data('source.csv', 'target.csv')
        self.assertIsInstance(source_df, pd.DataFrame)
        self.assertIsInstance(target_df, pd.DataFrame)

    def test_preprocess(self):
        text = "This is a sample text for preprocessing."
        processed_text = self.preprocessor.preprocess(text)
        self.assertIsInstance(processed_text, str)
        self.assertNotIn('this', processed_text.lower())
        self.assertNotIn('is', processed_text.lower())

    def test_preprocess_dataframes(self):
        source_df, target_df = self.preprocessor.preprocess_dataframes(self.source_df, self.target_df)
        self.assertIn('processed_category', source_df.columns)
        self.assertIn('processed_category', target_df.columns)
        self.assertEqual(len(source_df['processed_category']), len(self.source_df))
        self.assertEqual(len(target_df['processed_category']), len(self.target_df))

if __name__ == '__main__':
    unittest.main()
