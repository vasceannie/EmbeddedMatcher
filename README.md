# Category Matching and Scoring System

## Overview

This codebase provides a sophisticated solution for matching and scoring product categories across different classification systems. It's designed to help businesses and organizations streamline their product categorization processes, making it easier to compare and align categories from various sources using advanced natural language processing techniques.

## Key Components

### 1. Data Preprocessing (`src/data_preprocessing.py`)

The `DataPreprocessor` class handles the initial data preparation:

- Loads data from CSV files for both source and target categories.
- Supports hierarchical category structures in the target data.
- Performs text normalization:
  - Converts text to lowercase
  - Removes punctuation
  - Tokenizes the text
  - Removes stopwords
  - Applies lemmatization to reduce words to their base form

### 2. Synonym Matching (`src/synonym_matching.py`)

The `SynonymMatcher` class identifies categories with similar meanings:

- Utilizes WordNet to find synonyms, hypernyms, hyponyms, and similar terms for each word in a category.
- Calculates similarity scores based on shared synonyms between source and target categories.
- Applies a customizable threshold to determine valid matches.

### 3. Cosine Similarity Matching (`src/cosine_similarity.py`)

The `CosineSimilarityMatcher` class measures the textual similarity between categories:

- Generates BERT embeddings for each category using a pre-trained BERT model.
- Computes cosine similarity between source and target category embeddings.
- Identifies the best matching target category for each source category based on embedding similarity.

### 4. Combined Scoring (`src/scoring.py`)

The `CombinedMatcher` class merges results from synonym matching and cosine similarity:

- Combines scores from both matching techniques to determine the best overall match.
- Provides a final match and combined score for each source category.

### 5. Embedding Management (`src/utils.py`)

The `EmbeddingManager` class handles storage and retrieval of BERT embeddings:

- Supports saving and loading embeddings to/from various storage options:
  - CSV files
  - PostgreSQL databases
  - MongoDB databases
  - (Placeholder for universal database support)

### 6. Validation Metrics (`src/validation_metrics.py`)

The `ValidationMetrics` class calculates performance metrics:

- Computes precision, recall, and F1 score for the matching results.
- Provides methods to print the calculated metrics.

## How It Works

1. **Data Loading**: The system reads source and target categories from CSV files.

2. **Preprocessing**: Categories are cleaned and normalized to ensure fair comparisons.

3. **Synonym Matching**: Categories are compared based on shared synonyms and related terms.

4. **Cosine Similarity**: BERT embeddings are generated for each category, and similarity is measured using cosine distance.

5. **Combined Scoring**: Results from synonym matching and cosine similarity are merged to determine the best overall matches.

6. **Output**: The system provides a list of source categories with their best matches from the target list, along with confidence scores.

7. **Validation**: Performance metrics can be calculated to assess the quality of the matches.

## Getting Started

To use this system, you'll need:

1. Python 3.11 or later
2. Required libraries (listed in `pyproject.toml`)
3. Your source and target category lists in CSV format

### Setup

1. Install the required dependencies using Poetry:
   ```
   poetry install
   ```

2. Prepare your source and target CSV files with the appropriate column names:
   - Source file should have a 'classification_name' column
   - Target file should have either a 'category_name' column or hierarchical 'LvX_category_name' columns

3. Set up environment variables for database connections if using PostgreSQL or MongoDB (see `.env.example` for required variables)

### Usage

The main execution flow is not yet implemented in `src/main.py`, but the individual components can be used as follows:

1. Preprocess your data using `DataPreprocessor`
2. Perform synonym matching with `SynonymMatcher`
3. Generate embeddings and perform cosine similarity matching using `CosineSimilarityMatcher`
4. Combine the results using `CombinedMatcher`
5. Save or load embeddings as needed with `EmbeddingManager`
6. Calculate validation metrics using `ValidationMetrics`

## Technical Notes

- This system requires some technical knowledge to set up and run. If you're not comfortable with Python programming, consider seeking assistance from a developer or data scientist.
- The system uses BERT for generating embeddings, which may require significant computational resources for large datasets.
- Customization options are available for various parameters, such as similarity thresholds and embedding models.

## Future Improvements

- Implement the main execution flow in `src/main.py`
- Add support for additional languages beyond English
- Implement the universal database support in `EmbeddingManager`
- Develop a user interface for easier interaction with the system