# Machine Learning Models on Random Forest & Logistic Regression Algorithma

This repository contains two machine learning models implemented to classify products based on their nutritional value and harmfulness. The models analyze product ingredients and predict a classification level (A, B, C, D, E) based on the provided dataset.

## Models Overview

### Random Forest Classifier

1. A tree-based ensemble learning method.

2. Performs well with complex, non-linear data.

3. Uses multiple decision trees to enhance prediction accuracy.

### Logistic Regression

1. A statistical model for binary and multi-class classification.

2. Performs best with linear separable data.

3. Provides probabilistic outputs for predictions.

## Dataset

The dataset consists of 750 custom entries, each containing:

1. Product Name

2. Main Ingredients

3. Barcode Number

4. Nutritional Classification (Labels: A, B, C, D, E)


The dataset is used to train and validate the models to predict a product’s harmfulness level based on its ingredients.

## Dependencies

The following Python libraries are required to run the models:

numpy

pandas

scikit-learn

matplotlib

seaborn

## Training & Evaluation

### Random Forest Model

Initialization: The Random Forest model is initialized with 100 estimators and class weight balancing.

Training: The model is trained on the processed dataset.

Prediction: Predictions are generated for the test data.

Evaluation: The model is evaluated based on accuracy and other metrics.

### Logistic Regression Model

Initialization: The Logistic Regression model is initialized with default parameters and class weight balancing.

Training: The model is trained on the same dataset as Random Forest.

Prediction: Predictions are generated for the test data.

Evaluation: The model’s performance is compared with the Random Forest model.

## Comparison

The two models are compared based on the following:

Accuracy

Precision

Recall

F1 Score

Graphs are generated to visually represent the model’s performance.

File Structure

|-- random_forest_model.ipynb    	# Random Forest implementation
|-- logistic_regression_model.ipynb  	# Logistic Regression implementation
|-- dataset.csv                  	# Custom dataset
|-- comparison_notebook.ipynb    	# Model comparison notebook
|-- graphs/                      	# Contains performance graphs

## Usage

1. Clone the repository.

2. Install the required dependencies.

3. Run the respective model notebooks (NutriScore - Random Forest.ipynb and NutriScore - Logistic Regression.ipynb) to train and test the models.


## Results

The results of the model performance are saved in the graphs/ directory and displayed in the comparison notebook.

## Future Work

Expand the dataset to include more diverse products.

Experiment with additional classification models (e.g., SVM, Neural Networks).

Optimize the models further for better accuracy and generalization.
