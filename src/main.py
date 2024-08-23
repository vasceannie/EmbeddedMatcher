import pandas as pd
from src.validation_metrics import ValidationMetrics

# ... (your existing import statements and code)

def main():
    # ... (your existing code for loading and processing data)

    # Assuming you have a DataFrame called 'source_df' with 'true_category' and 'final_match' columns
    true_labels = source_df['true_category']
    predicted_labels = source_df['final_match'].apply(lambda x: x[0] if x else None)

    # Calculate metrics
    metrics = ValidationMetrics.calculate_metrics(true_labels, predicted_labels)

    # Print metrics
    ValidationMetrics.print_metrics(metrics)

    # ... (rest of your code)

if __name__ == "__main__":
    main()