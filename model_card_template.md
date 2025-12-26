# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model used is a Random Forest Classifier from the package scikit-learn. The hyperparameters were chosen based on a grid based search over the following parameters:
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
The hyperparameters were validated using 5 cross folds on the training data. Code can be found within ml/model.py

## Intended Use

The model has been trained to classify whether a person living in the United States makes over 50K a year, using census data as indicators. 

## Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 

The original data set has 48842 rows, and a 80% was used for training. The data was processed using an One Hot Encoder for the categorical variables and Label Binarizer for the response variable salary. 

## Evaluation Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 

The original data set has 48842 rows, and a 20% was used for training. The data was processed using an One Hot Encoder for the categorical variables and Label Binarizer for the response variable salary. 

## Metrics

The model was optimized for the F1 score. The value on the test set of the F1 score is 0.69, the recall is 0.64 and the precision is 0.74.

## Ethical Considerations

The model discriminates on race, gender and origin country. 

## Caveats and Recommendations

The data comes from the 1994 census database and it is no longer representative of the current demographics of the US. I recommend this model should not be used in production due to the aforementioned model drift as well as due to ethical considerations. 
