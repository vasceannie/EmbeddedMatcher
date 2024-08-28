# Category Matching and Scoring System

## Overview

This advanced system provides a sophisticated solution for matching and scoring product categories across different classification systems. It's designed to help businesses and organizations streamline their product categorization processes, making it easier to compare and align categories from various sources using state-of-the-art natural language processing techniques.

## Key Features

- Data preprocessing and normalization
- Synonym-based matching using WordNet
- BERT-based embedding generation for semantic understanding
- Cosine similarity matching for high-precision category alignment
- Combined scoring mechanism for optimal match selection
- Flexible embedding storage and retrieval options
- Performance validation metrics

## Components

### 1. Data Preprocessing (`src/data_preprocessing.py`)

The `DataPreprocessor` class handles initial data preparation:

- Loads data from CSV files for both source and target categories
- Supports hierarchical category structures in target data
- Performs text normalization:
  - Converts text to lowercase
  - Removes punctuation and extra whitespace
  - Tokenizes the text
  - Removes stopwords
  - Applies lemmatization

### 2. Synonym Matching (`src/synonym_matching.py`)

The `SynonymMatcher` class identifies categories with similar meanings:

- Utilizes WordNet to find synonyms, hypernyms, and hyponyms
- Calculates similarity scores based on shared synonyms
- Applies a customizable threshold for match determination

### 3. Cosine Similarity Matching (`src/cosine_similarity.py`)

The `CosineSimilarityMatcher` class measures textual similarity between categories:

- Generates BERT embeddings for each category
- Computes cosine similarity between embeddings
- Identifies best matching target categories based on embedding similarity

### 4. Combined Scoring (`src/scoring.py`)

The `CombinedMatcher` class merges results from synonym and cosine similarity matching:

- Combines scores using configurable weights
- Determines the best overall match for each source category

### 5. Embedding Management (`src/utils.py`)

The `EmbeddingManager` class handles BERT embedding storage and retrieval:

- Supports multiple storage options:
  - CSV files
  - PostgreSQL databases
  - MongoDB databases
- Provides serialization and deserialization methods for complex data types

### 6. Validation Metrics (`src/validation_metrics.py`)

The `ValidationMetrics` class calculates performance metrics:

- Computes precision, recall, and F1 score
- Provides formatted output of calculated metrics

## Installation

1. Ensure you have Python 3.11 or later installed.
2. Clone this repository:
   ```
   git clone https://github.com/your-username/category-matching-system.git
   cd category-matching-system
   ```
3. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

1. Prepare your source and target CSV files:
   - Source file should have a 'classification_name' column
   - Target file should have either a 'category_name' column or hierarchical 'LvX_category_name' columns

2. Set up environment variables for database connections if using PostgreSQL or MongoDB (refer to `.env.example`).

3. Run the main script:
   ```
   python main.py
   ```

4. The script will perform the following steps:
   - Load and preprocess data
   - Apply synonym matching
   - Generate BERT embeddings
   - Perform cosine similarity matching
   - Combine scores and determine best matches
   - Save results and embeddings
   - Calculate and display validation metrics

5. Results will be saved in the `matched_results.csv` file.

## Configuration

You can adjust various parameters in the code:

- `SynonymMatcher`: Modify the similarity threshold in the constructor
- `CombinedMatcher`: Adjust weights for synonym and cosine scores
- `CosineSimilarityMatcher`: Change the BERT model or adjust embedding parameters

## Performance Considerations

- BERT embedding generation can be computationally intensive for large datasets. Consider using GPU acceleration if available.
- For very large datasets, you may need to process data in batches to manage memory usage.

## Testing

Run the test suite using:
```python -m unittest discover tests```


## Future Improvements

- Implement support for additional languages
- Develop a user interface for easier interaction
- Add support for more embedding models (e.g., RoBERTa, ALBERT)
- Implement distributed processing for large-scale datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and WordNet for synonym matching capabilities
- Hugging Face Transformers for BERT implementation
- scikit-learn for cosine similarity and validation metrics