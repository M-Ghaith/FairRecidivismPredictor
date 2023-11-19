from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

def statistical_parity(y_test, y_pred, feature):
    """
    Calculate statistical parity metrics.

    This function computes the statistical parity difference and ratio,
    which are measures of fairness in terms of demographic parity.

    Parameters:
    y_test (pd DataFrame): True labels for the test data.
    y_pred (pd Serie): Predicted labels for the test data.
    feature (pd Serie): Sensitive feature based on which fairness is evaluated.

    Returns:
    dict: A dictionary containing the Statistical Parity Difference and Ratio.
    """

    stat_parity_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=feature)
    stat_parity_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=feature)
    
    return {
        "Statistical Parity Difference": round(stat_parity_diff, 3),
        "Statistical Parity ratio": round(stat_parity_ratio, 3)
    }

def equalized_odds(y_test, y_pred, feature):
    """
    Calculate equalized odds metrics.

    This function computes the equalized odds difference and ratio,
    which are measures of fairness in terms of error rates.

    Parameters:
    y_test (pd DataFrame): True labels for the test data.
    y_pred (pd Serie): Predicted labels for the test data.
    feature (pd Serie): Sensitive feature based on which fairness is evaluated.

    Returns:
    dict: A dictionary containing the Equalized Odds Difference and Ratio.
    """
    
    equal_odds_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=feature)
    equal_odds_ratio = equalized_odds_ratio(y_test, y_pred, sensitive_features=feature)
    
    return {
        "Equalized Odds Difference": round(equal_odds_diff, 3),
        "Equalized Odds Ratio": round(equal_odds_ratio, 3)
    }