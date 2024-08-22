import sys
import os
import numpy as np

# Add the parent directory to the system path to allow importing from the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
from src.cosine_similarity import CosineSimilarityMatcher

class TestCosineSimilarityMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = CosineSimilarityMatcher()

    def test_get_bert_embedding(self):
        text = "This is a test sentence"
        embedding = self.matcher.get_bert_embedding(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (768,))  # BERT base model output dimension

    def test_generate_embeddings(self):
        df = pd.DataFrame({
            'processed_category': ['car repair', 'book store']
        })
        result_df = self.matcher.generate_embeddings(df)
        
        self.assertIn('bert_embedding', result_df.columns)
        self.assertEqual(len(result_df['bert_embedding']), 2)
        self.assertIsInstance(result_df['bert_embedding'][0], np.ndarray)
        self.assertEqual(result_df['bert_embedding'][0].shape, (768,))

    def test_cosine_match(self):
        source_embedding = np.random.rand(768)
        target_embeddings = [np.random.rand(768) for _ in range(3)]
        
        best_match, similarity = self.matcher.cosine_match(source_embedding, target_embeddings)
        
        self.assertIsInstance(best_match, np.ndarray)
        self.assertEqual(best_match.shape, (768,))
        self.assertIsInstance(similarity, float)
        self.assertTrue(0 <= similarity <= 1)

    def test_apply_cosine_matching(self):
        source_df = pd.DataFrame({
            'processed_category': ['car repair', 'book store'],
            'bert_embedding': [np.random.rand(768), np.random.rand(768)]
        })
        target_df = pd.DataFrame({
            'processed_category': ['auto fix', 'library', 'vehicle service'],
            'bert_embedding': [np.random.rand(768), np.random.rand(768), np.random.rand(768)]
        })
        
        result_df = self.matcher.apply_cosine_matching(source_df, target_df)
        
        self.assertIn('cosine_matches', result_df.columns)
        self.assertEqual(len(result_df['cosine_matches']), 2)
        self.assertIsInstance(result_df['cosine_matches'][0], tuple)
        self.assertEqual(len(result_df['cosine_matches'][0]), 2)  # (best_match, similarity_score)
        self.assertEqual(result_df['cosine_matches'][0][0].shape, (768,))

if __name__ == '__main__':
    unittest.main()