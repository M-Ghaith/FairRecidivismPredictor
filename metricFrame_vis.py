# Importing necessary modules from Fairlearn and sklearn for fairness analysis and evaluation
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import balanced_accuracy_score, precision_score

# Function to perform fairness analysis using Fairlearn's MetricFrame
def analyze_metrics_using_metricFrame(y_test, y_pred, feature):
    """
    Analyzes and visualizes various fairness metrics using Fairlearn's MetricFrame.

    Parameters:
    y_test (pd serie): The true labels of the test data.
    y_pred (pd serie): The predicted labels from the model.
    feature (pd serie): Sensitive feature(s) to analyze for fairness (e.g., race, gender).

    Returns:
    matplotlib.axes.Axes: A set of bar plots for each group within the sensitive feature, showing the performance across different metrics.
    """

    # Defining a dictionary of metrics to analyze
    metrics = {
        "accuracy": balanced_accuracy_score,  # balanced accuracy computes accuracy on a balanced dataset
        "precision": precision_score,  # precision measures the accuracy of positive predictions
        "false positive rate": false_positive_rate,  # false positive rate for each group
        "false negative rate": false_negative_rate,  # false negative rate for each group
        "selection rate": selection_rate,  # proportion of positive predictions, indicating potential bias
    }

    # Creating a MetricFrame object to evaluate the metrics across different groups
    metric_frame = MetricFrame(
        metrics=metrics,  # the metrics to compute
        y_true=y_test,  # the true labels
        y_pred=y_pred,  # the predicted labels
        sensitive_features=feature  # the sensitive feature for fairness analysis
    )

    # Plotting the metrics for each group in a bar plot format
    return metric_frame.by_group.plot.bar(
        subplots=True,       # creates a separate plot for each metric
        layout=[3, 3],       # layout of the subplots
        legend=False,        # no legend for clarity
        figsize=[16, 10],    # size of the overall figure
        ylim=[0, 1.05],      # y-axis limits
        title="Show all metrics",  # title of the plot
    )
