from sklearn.metrics import precision_score, recall_score, f1_score

class ValidationMetrics:
    """
    A class to calculate and print validation metrics for classification results.

    This class provides static methods to calculate precision, recall, and F1 score
    for given true and predicted labels, and to print these metrics in a formatted manner.
    """

    @staticmethod
    def calculate_metrics(true_labels, predicted_labels):
        """
        Calculate precision, recall, and F1 score for the matching results.

        This method uses the sklearn library to calculate the precision, recall, and F1 score
        based on the provided true and predicted labels. The scores are calculated using the
        'weighted' average method, which accounts for label imbalance.

        Args:
            true_labels (list): The ground truth category labels.
            predicted_labels (list): The predicted category labels.

        Returns:
            dict: A dictionary containing the precision, recall, and F1 score.
        """
        # Calculate the precision score using sklearn's precision_score function
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        
        # Calculate the recall score using sklearn's recall_score function
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        
        # Calculate the F1 score using sklearn's f1_score function
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Return the calculated metrics as a dictionary
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    @staticmethod
    def print_metrics(metrics):
        """
        Print the calculated metrics.

        This method prints the precision, recall, and F1 score in a formatted manner.
        Each metric is printed with four decimal places for better readability.

        Args:
            metrics (dict): A dictionary containing the precision, recall, and F1 score.
        """
        # Print the precision score with four decimal places
        print(f"Precision: {metrics['precision']:.4f}")
        
        # Print the recall score with four decimal places
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Print the F1 score with four decimal places
        print(f"F1 Score: {metrics['f1_score']:.4f}")