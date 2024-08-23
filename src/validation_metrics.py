from sklearn.metrics import precision_score, recall_score, f1_score

class ValidationMetrics:
    @staticmethod
    def calculate_metrics(true_labels, predicted_labels):
        """
        Calculate precision, recall, and F1 score for the matching results.

        Args:
            true_labels (list): The ground truth category labels.
            predicted_labels (list): The predicted category labels.

        Returns:
            dict: A dictionary containing the precision, recall, and F1 score.
        """
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    @staticmethod
    def print_metrics(metrics):
        """
        Print the calculated metrics.

        Args:
            metrics (dict): A dictionary containing the precision, recall, and F1 score.
        """
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")