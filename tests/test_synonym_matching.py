import sys
import os

# Add the parent directory to the system path to allow importing from the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from src.synonym_matching import SynonymMatcher

class TestSynonymMatcher(unittest.TestCase):
    """
    Unit test class for testing the SynonymMatcher functionality.
    This class contains test cases to verify the behavior of the
    SynonymMatcher methods, including synonym retrieval and matching.
    """

    def setUp(self):
        """
        Set up the test environment by initializing a SynonymMatcher instance
        with a default threshold for similarity matching.
        """
        self.matcher = SynonymMatcher(threshold=0.1)  # Lowered threshold for testing

    def test_get_synonyms(self):
        """
        Test the get_synonyms method of the SynonymMatcher class.
        This test checks if the method correctly retrieves synonyms for a given word.
        """
        synonyms = self.matcher.get_synonyms("car")
        # Assert that at least one of the expected synonyms is present in the retrieved synonyms
        self.assertTrue(any(syn in synonyms for syn in ["automobile", "vehicle", "auto"]))

    def test_synonym_match(self):
        source_category = "car repair"
        target_categories = pd.DataFrame({
            'category_name': ['auto fix', 'book store', 'vehicle service'],
            'processed_category': ['auto fix', 'book store', 'vehicle service']
        })
        
        matches = self.matcher.synonym_match(source_category, target_categories)
        
        self.assertIsInstance(matches, list)
        self.assertTrue(len(matches) > 0)
        if len(matches) > 0:
            self.assertIsInstance(matches[0], tuple)
            self.assertEqual(len(matches[0]), 2)
            self.assertIn(matches[0][0], target_categories['category_name'].values)
            self.assertIsInstance(matches[0][1], float)
            self.assertTrue(self.matcher.threshold < matches[0][1] <= 1)

    def test_apply_synonym_matching(self):
        """
        Test the apply_synonym_matching method of the SynonymMatcher class.
        This test checks if the method correctly applies synonym matching
        to a DataFrame of source categories against a DataFrame of target categories.
        """
        source_df = pd.DataFrame({
            'category': ['car repair', 'book shop'],
            'processed_category': ['car repair', 'book shop']
        })
        target_df = pd.DataFrame({
            'category': ['auto fix', 'library', 'vehicle service'],
            'processed_category': ['auto fix', 'library', 'vehicle service']
        })
        
        result_df = self.matcher.apply_synonym_matching(source_df, target_df)
        
        self.assertIn('synonym_matches', result_df.columns)
        self.assertGreaterEqual(len(result_df['synonym_matches'][0]), 1)  # 'car repair' should match at least one
        self.assertEqual(len(result_df['synonym_matches'][1]), 0)  # 'book shop' shouldn't match any

if __name__ == '__main__':
    unittest.main()