import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.synonym_matching import SynonymMatcher
from src.cosine_similarity import CosineSimilarityMatcher
from src.scoring import CombinedMatcher
from src.utils import EmbeddingManager
from src.validation_metrics import ValidationMetrics

def main():
    """
    Main function to execute the category matching and scoring process.
    """
    # Step 1: Data Loading and Preprocessing
    data_preprocessor = DataPreprocessor()
    source_df, target_df = data_preprocessor.load_data('distinct_source_class.csv', 'target_taxonomy.csv')
    source_df, target_df = data_preprocessor.preprocess_dataframes(source_df, target_df)

    # Print column names after preprocessing
    print("Source DataFrame columns after preprocessing:", source_df.columns.tolist())
    print("Target DataFrame columns after preprocessing:", target_df.columns.tolist())

    # Step 2: Synonym Matching
    synonym_matcher = SynonymMatcher()
    source_df = synonym_matcher.apply_synonym_matching(source_df, target_df)

    # Step 3: Cosine Similarity Matching
    cosine_matcher = CosineSimilarityMatcher()
    source_df = cosine_matcher.generate_embeddings(source_df)
    target_df = cosine_matcher.generate_embeddings(target_df)
    source_df = cosine_matcher.apply_cosine_matching(source_df, target_df)

    # Step 4: Combined Scoring
    combined_matcher = CombinedMatcher()
    source_df = combined_matcher.apply_combined_scoring(source_df)

    # Step 5: Output
    print("Matching Results:")
    for _, row in source_df.iterrows():
        print(f"Source: {row['category']}, Best Match: {row['final_match'][0]}, Score: {row['final_match'][1]:.2f}")

    # Optional: Save embeddings
    embedding_manager = EmbeddingManager()
    embedding_manager.save_embeddings(source_df[['category', 'bert_embedding']], 'source_embeddings.csv')
    embedding_manager.save_embeddings(target_df[['category', 'bert_embedding']], 'target_embeddings.csv')

    # Step 6: Validation
    true_labels = source_df['category']
    predicted_labels = source_df['final_match'].apply(lambda x: x[0])
    metrics = ValidationMetrics.calculate_metrics(true_labels, predicted_labels)
    ValidationMetrics.print_metrics(metrics)

    # Save final results
    source_df.to_csv('matched_results.csv', index=False)

if __name__ == "__main__":
    main()