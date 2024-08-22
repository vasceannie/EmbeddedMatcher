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
        self.matcher = SynonymMatcher(threshold=0.3)

    def test_get_synonyms(self):
        """
        Test the get_synonyms method of the SynonymMatcher class.
        This test checks if the method correctly retrieves synonyms for a given word.
        """
        synonyms = self.matcher.get_synonyms("car")
        # Assert that at least one of the expected synonyms is present in the retrieved synonyms
        self.assertTrue(any(syn in synonyms for syn in ["automobile", "vehicle", "auto"]))

    def test_synonym_match(self):
        """
        Test the synonym_match method of the SynonymMatcher class.
        This test verifies that the method correctly matches a source category
        against a set of target categories based on shared synonyms.
        """
        source_category = "automobile repair"
        target_categories = pd.DataFrame({
            'category': ['car fix', 'book store', 'vehicle maintenance'],
            'processed_category': ['car fix', 'book store', 'vehicle maintenance']
        })
        
        # Perform synonym matching
        matches = self.matcher.synonym_match(source_category, target_categories)
        print(f"Matches for 'automobile repair': {matches}")  # Debug print
        
        # Assert that at least one match is found
        self.assertGreaterEqual(len(matches), 1)
        # Assert that the matches include expected target categories
        self.assertTrue(any(match[0] in ['car fix', 'vehicle maintenance'] for match in matches))

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
        
        # Apply synonym matching to the source DataFrame
        result_df = self.matcher.apply_synonym_matching(source_df, target_df)
        print(f"Result DataFrame: {result_df}")  # Debug print
        
        # Assert that the result DataFrame contains the 'synonym_matches' column
        self.assertIn('synonym_matches', result_df.columns)
        # Assert that 'car repair' matches at least one target category
        self.assertGreaterEqual(len(result_df['synonym_matches'][0]), 1)
        # Assert that 'book shop' doesn't match any target category
        self.assertEqual(len(result_df['synonym_matches'][1]), 0)

if __name__ == '__main__':
    unittest.main()