# ML

Machine Learning Model Comparison Toolkit

A comprehensive Python package for automatically comparing and evaluating 25+ machine learning classifiers with parallel execution.

## ✨ Features
- 25+ classification algorithms
- Parallel execution for fast performance
- Comprehensive evaluation metrics
- Cross-validation support
- Easy to use


from model_comparison import compare_ml_models

from sklearn.datasets import load_iris


# Load data

data = load_iris()

X, y = data.data, data.target

# Compare models

results = compare_ml_models(X, y)

print(results)

X (Features):

Shape: (150, 4)

Contains: Numerical measurements of iris flowers

Columns: ['sepal length', 'sepal width', 'petal length', 'petal width']

y (Target):

Shape: (150,)

Contains: Class labels (0, 1, 2 representing different iris species)

Values: 0='setosa', 1='versicolor', 2='virginica'