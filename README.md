# ECE 570 Final Project 

This research investigates the efficacy of Shapley-based data and fairness debugging techniques across complete machine learning pipelines. I extend existing work on training data debugging by incorporating both model performance optimization on accuracy and algorithmic fairness constraints. This approach quantifies the influence of individual training data points throughout the entire ML pipeline, from preprocessing to prediction, and identifies which data points contribute most significantly to both accuracy degradation and fairness violations.

## Code Structure

1. **Data_Debugging_On_UCI_and_Titanic.ipynb**
    Comparative analysis of data cleaning techniques
    Focuses on data debugging experiments on:
    * UCI Adult Income dataset
    * Titanic dataset

2. **Fairness_Debugging_on_UCI.ipynb**

    Fairness debugging on the UCI Adult Income dataset by conducting Equalized odds evaluation



3. **German_Credit_Using_MLP_and_GB.ipynb**
    Experiments with German Credit dataset on both fairness and data debugging
    Uses multiple machine learning models:
    * Multi-Layer Perceptron (MLP)
    * Gradient Boosting (GB)

# Experimental Setup
## Datasets

1. **UCI Adult Income Dataset**:

    48,842 instances with 14 attributes
    Binary classification: income >$50K (1) vs. ≤$50K (0)
    Sensitive attributes: sex, race, and age
    Features include: education, occupation, work class, hours-per-week, etc.
    Feature types: 6 numerical, 8 categorical

2. **German Credit Risk Dataset**:

    1,000 instances with 20 attributes
    Binary classification: good (1) vs. bad (0) credit risk
    Sensitive attributes: gender (derived from personal status) and age
    Features include: account status, credit history, loan purpose, employment, etc.
    Feature types: 7 numerical, 13 categorical

3. **Titanic Survival Dataset**:

    891 instances with 12 attributes
    Binary classification: survived (1) vs. did not survive (0)
    Sensitive attributes: gender and passenger class
    Features include: age, fare, cabin, embarkation point, etc.
    Feature types: 5 numerical, 7 categorical

## Methodology

**Data Preparation:**
Train/validation/test split: 60%/20%/20%
Controlled noise injection: 20% of training labels flipped randomly
Validation set used for importance score computation

**Importance Computation:**

1. Utility functions:

    SklearnModelAccuracy
    SklearnModelEqualizedOddsDifference

2. Label Cleaning Experiments:

    Noise level: 20%
    Cleaning strategies: Shapley-guided vs. random selection
    Cleaning percentages: 0%, 5%, 10%, 20%, 30%, 50%, 100%

3. Fairness Analysis:

    Demographic breakdown of influential points
    Comparison of accuracy-focused vs. fairness-focused importance scores

## Key Findings
**Data Debugging**

* Shapley-based cleaning outperformed random cleaning
* Most datasets showed significant performance improvements
* Some datasets (like Titanic) revealed limitations of the approach
* Not all data points contribute equally to model performance

**Fairness Debugging**

* Most training samples have negligible individual impact on fairness
* A small number of points can have disproportionate influence
* DataScope-guided cleaning consistently showed more efficient fairness improvements

## Acknowledgments
This research builds upon the Datascope framework and leverages libraries including scikit-learn, fairlearn, numpy, and pandas. Computations were performed on Google Colab.