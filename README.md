# COMPAS Recidivism Prediction Fairness Analysis
This repository contains a series of Jupyter notebooks and Python scripts aimed at assessing the fairness of machine learning models used to predict recidivism. The analysis is inspired by the study of the COMPAS tool, which showed disparities in prediction outcomes across different races and genders.

## Repository Structure
### Notebooks

**1. EDA_analysis.ipynb**
Conducts exploratory data analysis (EDA) to assess data completeness, validity, and distribution across features such as age, race, and gender.
Performs correlation analysis and comparative analysis by race and gender.

**2. ML_models.ipynb**
Develops machine learning models to predict recidivism, establishes a performance baseline, and tests multiple models using nested cross-validation.
Evaluates the top two models using repeated cross-validation and tunes them using GridSearchCV.
Assesses model performance using metrics like AUC, accuracy, precision, recall, and F1 score.

**3. model_fairness_analysis.ipynb**
Focuses on fairness analysis by examining the impact of removing the sensitive features 'race' and 'sex' on model predictions and fairness metrics.
Applies class resampling techniques and in-processing techniques like data reweighting and exponentiated gradients to mitigate bias.
Uses global and local explainability methods to interpret model predictions and investigates model safety and uncertainty estimation.
This experiment optimizes fairness for both 'race' and 'sex'.

**4. model_fairness_race_analysis.ipynb**
Focuses on fairness analysis by examining the impact of removing the sensitive 'race' and 'sex' features on model predictions and fairness metrics.
Applies class resampling techniques and in-processing techniques like data reweighting and exponentiated gradients to mitigate bias.
Uses global and local explainability methods to interpret model predictions and investigates model safety and uncertainty estimation.
This is a rerun of the previous experiment that optimizes fairness for 'race' only and achieves better scores.

**5. resampling_analysis.ipynb**
Investigates which resampling technique results in a balance of fairness across metrics of statistical parity and equalized odds.

**6. resampling_race_analysis.ipynb**
Investigates which resampling technique results in a balance of fairness across metrics of statistical parity and equalized odds.
This is a rerun of the previous experiment that includes only the 'race' feature, rather than both 'race' and 'sex'.

### Scripts

**7. fairness_metrics.py**
Defines a function to calculate statistical parity difference and ratio as measures of demographic parity.

**8. metricFrame_vis.py**
Provides functionality to analyze and visualize fairness metrics using Fairlearn's MetricFrame.

## Introduction

The analysis begins with an in-depth exploration of the dataset to ensure its integrity and to understand the initial distribution of features. Subsequent notebooks develop predictive models and thoroughly test their performance and fairness. The fairness analysis is particularly focused on addressing and mitigating biases that were identified in the COMPAS analysis, ensuring that our models do not perpetuate these disparities.

### Fairness Analysis
Each model's fairness is evaluated in the context of its predictions on race and sex. We apply various techniques to assess and improve fairness metrics, including statistical parity and equalized odds. The process involves not only technical adjustments such as resampling and reweighting but also a careful examination of model predictions to ensure ethical application.

### Model Interpretability and Safety
The repository also emphasizes the importance of model interpretability and safety. Techniques like LIME and feature importance analysis are used to make the model's decisions transparent and understandable. Furthermore, we estimate the epistemic uncertainty to gauge the confidence in the model's predictions and to identify areas where caution is warranted.

### Conclusion
This comprehensive analysis aims to contribute to the development of fairer predictive models in criminal justice. We combine rigorous data science techniques with commitment to ethical standards to improve decision-making processes and outcomes.
