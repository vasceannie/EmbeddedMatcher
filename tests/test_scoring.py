import sys
import os

# Add the parent directory to the system path to allow importing from the src module
# This is necessary for the test cases to access the CombinedMatcher class from the scoring module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from src.scoring import CombinedMatcher

class TestCombinedMatcher(unittest.TestCase):
    """
    Unit test class for testing the CombinedMatcher functionality.

    This class contains test cases to verify the behavior of the CombinedMatcher methods,
    specifically the combined matching of synonym and cosine similarity results.
    """

    def setUp(self):
        """
        Set up the test environment by initializing sample synonym and cosine matches.

        This method is called before each test case. It initializes two lists:
        - synonym_matches: A list of tuples representing synonym matches and their scores.
        - cosine_matches: A list of tuples representing cosine matches and their scores.
        """
        self.synonym_matches = [('car fix', 0.8), ('vehicle maintenance', 0.6)]
        self.cosine_matches = [('auto fix', 0.9), ('library', 0.4)]

    def test_combined_match_with_both_matches(self):
        """
        Test the combined_match method when both synonym and cosine matches are provided.

        This test verifies that the method correctly identifies the best match and its combined score
        when both types of matches are available.
        """
        best_match, combined_score = CombinedMatcher.combined_match(self.synonym_matches, self.cosine_matches)
        self.assertEqual(best_match, 'auto fix')  # Check if the best match is as expected
        self.assertAlmostEqual(combined_score, 0.9)  # Check if the combined score is approximately correct

    def test_combined_match_with_only_synonym_matches(self):
        """
        Test the combined_match method when only synonym matches are provided.

        This test verifies that the method correctly identifies the best match and its score
        when only synonym matches are available.
        """
        best_match, combined_score = CombinedMatcher.combined_match(self.synonym_matches, [])
        self.assertEqual(best_match, 'car fix')  # Check if the best match is as expected
        self.assertEqual(combined_score, 0.8)  # Check if the score is as expected

    def test_combined_match_with_only_cosine_matches(self):
        """
        Test the combined_match method when only cosine matches are provided.

        This test verifies that the method correctly identifies the best match and its score
        when only cosine matches are available.
        """
        best_match, combined_score = CombinedMatcher.combined_match([], self.cosine_matches)
        self.assertEqual(best_match, 'auto fix')  # Check if the best match is as expected
        self.assertEqual(combined_score, 0.9)  # Check if the score is as expected

    def test_combined_match_with_no_matches(self):
        """
        Test the combined_match method when no matches are provided.

        This test verifies that the method returns None for the best match and a score of 0
        when no matches are available.
        """
        best_match, combined_score = CombinedMatcher.combined_match([], [])
        self.assertIsNone(best_match)  # Check if the best match is None
        self.assertEqual(combined_score, 0)  # Check if the score is 0

    def test_apply_combined_scoring(self):
        """
        Test the apply_combined_scoring method to ensure it processes the DataFrame correctly.

        This test verifies that the method adds a 'final_match' column to the DataFrame
        and that the values in this column are as expected based on the provided matches.
        """
        df = pd.DataFrame({
            'synonym_matches': [self.synonym_matches, []],
            'cosine_matches': [self.cosine_matches, self.cosine_matches]
        })
        result_df = CombinedMatcher.apply_combined_scoring(df)
        self.assertIn('final_match', result_df.columns)  # Check if 'final_match' column is added
        self.assertEqual(result_df['final_match'][0][0], 'auto fix')  # Check the first final match
        self.assertEqual(result_df['final_match'][1][0], 'auto fix')  # Check the second final match

if __name__ == '__main__':
    unittest.main()