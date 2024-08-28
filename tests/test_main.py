import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from main import (
    get_postgres_engine,
    save_checkpoint,
    load_checkpoint,
    load_source_data,
    load_target_data,
    main
)

class TestMain(unittest.TestCase):

    def setUp(self):
        self.mock_engine = create_engine('sqlite:///:memory:')

    @patch('main.create_engine')
    def test_get_postgres_engine(self, mock_create_engine):
        mock_create_engine.return_value = self.mock_engine
        engine = get_postgres_engine()
        self.assertEqual(engine, self.mock_engine)

    def test_save_checkpoint(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        })
        table_name = 'test_table'
        save_checkpoint(df, table_name, self.mock_engine)
        
        # Mock the pd.read_sql_table function
        with patch('pandas.read_sql_table') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['[1, 2]', '[3, 4]', '[5, 6]']
            })
            
            # Mock the inspect function to simulate the table existing
            with patch('main.inspect') as mock_inspect:
                mock_inspector = MagicMock()
                mock_inspect.return_value = mock_inspector
                mock_inspector.has_table.return_value = True
                
                saved_df = load_checkpoint(table_name)
        
        # Compare the original df with the loaded df
        pd.testing.assert_frame_equal(df, saved_df)

    @patch('main.get_postgres_engine')
    @patch('main.inspect')
    @patch('main.pd.read_sql_table')
    def test_load_checkpoint(self, mock_read_sql_table, mock_inspect, mock_get_postgres_engine):
        mock_engine = MagicMock()
        mock_get_postgres_engine.return_value = mock_engine
        
        # Test when table exists
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = True
        
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_sql_table.return_value = mock_df
        
        result = load_checkpoint('test_table')
        pd.testing.assert_frame_equal(result, mock_df)
        mock_read_sql_table.assert_called_once_with('test_table', mock_engine)
        
        # Test when table doesn't exist
        mock_inspector.has_table.return_value = False
        result = load_checkpoint('nonexistent_table')
        self.assertIsNone(result)
        
        # Reset mock to ensure it's clean for the next test
        mock_read_sql_table.reset_mock()

    @patch('main.pd.read_csv')
    def test_load_source_data(self, mock_read_csv):
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        result = load_source_data()
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('main.pd.read_csv')
    def test_load_target_data(self, mock_read_csv):
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        result = load_target_data()
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('main.CosineSimilarityMatcher')
    @patch('main.SynonymMatcher')
    @patch('main.DataPreprocessor')
    @patch('main.load_checkpoint')
    @patch('main.load_source_data')
    @patch('main.load_target_data')
    @patch('main.get_postgres_engine')
    @patch('main.save_checkpoint')
    @patch('main.CombinedMatcher')
    @patch('main.EmbeddingManager')
    @patch('main.ValidationMetrics')
    def test_main(self, mock_validation_metrics, mock_embedding_manager, mock_combined_matcher, 
                  mock_save_checkpoint, mock_get_postgres_engine, mock_load_target_data, 
                  mock_load_source_data, mock_load_checkpoint, mock_data_preprocessor, 
                  mock_synonym_matcher, mock_cosine_similarity_matcher):
        # Set up your mocks
        mock_engine = MagicMock()
        mock_get_postgres_engine.return_value = mock_engine
        
        mock_load_checkpoint.return_value = None  # Simulate no checkpoints found
        
        mock_source_df = pd.DataFrame({
            'classification_name': ['A', 'B'],
            'processed_category': ['a', 'b'],
            'synonym_matches': [[], []],
            'cosine_matches': [[], []],
            'final_match': [('X', 0.8), ('Y', 0.7)],
            'bert_embedding': [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        })
        mock_target_df = pd.DataFrame({
            'category': ['X', 'Y'], 
            'processed_category': ['x', 'y'],
            'bert_embedding': [np.array([0.5, 0.6]), np.array([0.7, 0.8])]
        })
        mock_load_source_data.return_value = mock_source_df
        mock_load_target_data.return_value = mock_target_df
        
        mock_preprocessor = MagicMock()
        mock_data_preprocessor.return_value = mock_preprocessor
        mock_preprocessor.preprocess_dataframes.return_value = (mock_source_df, mock_target_df)
        
        mock_synonym_matcher_instance = MagicMock()
        mock_synonym_matcher.return_value = mock_synonym_matcher_instance
        mock_synonym_matcher_instance.apply_synonym_matching.return_value = mock_source_df
        
        mock_cosine_matcher_instance = MagicMock()
        mock_cosine_similarity_matcher.return_value = mock_cosine_matcher_instance
        mock_cosine_matcher_instance.generate_embeddings.return_value = mock_source_df
        mock_cosine_matcher_instance.apply_cosine_matching.return_value = mock_source_df

        mock_combined_matcher_instance = MagicMock()
        mock_combined_matcher.return_value = mock_combined_matcher_instance
        mock_combined_matcher_instance.apply_combined_scoring.return_value = mock_source_df

        mock_embedding_manager_instance = MagicMock()
        mock_embedding_manager.return_value = mock_embedding_manager_instance

        # Call the main function
        main()

        # Modify the assertions to be more flexible
        mock_preprocessor.preprocess_dataframes.assert_called()
        mock_synonym_matcher_instance.apply_synonym_matching.assert_called()
        
        # Check if generate_embeddings was called with a DataFrame containing the expected columns
        calls = mock_cosine_matcher_instance.generate_embeddings.call_args_list
        self.assertEqual(len(calls), 2)  # Ensure it was called twice
        for call in calls:
            args, _ = call
            df = args[0]
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('processed_category', df.columns)
        
        mock_cosine_matcher_instance.apply_cosine_matching.assert_called()
        mock_combined_matcher_instance.apply_combined_scoring.assert_called()
        mock_embedding_manager_instance.save_embeddings.assert_called()
        mock_validation_metrics.calculate_metrics.assert_called()
        mock_validation_metrics.print_metrics.assert_called()
        
        # Assert that save_checkpoint was called the expected number of times
        self.assertEqual(mock_save_checkpoint.call_count, 4)

if __name__ == '__main__':
    unittest.main()