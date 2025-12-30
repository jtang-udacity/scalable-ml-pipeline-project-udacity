# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model type:** RandomForestClassifier trained via `GridSearchCV` (5-fold CV) from scikit-learn.
- **Training script:** `train_model.py` calls `ml/model.py::train_model`.
- **Hyperparameter search space:**
  - `n_estimators`: 200, 500
  - `max_features`: log2, sqrt
  - `max_depth`: 4, 5, 100
  - `criterion`: gini, entropy
- **Best model artifact:** saved as `model/model.pkl` (with corresponding encoder `model/encoder.pkl` and label binarizer `model/lb.pkl`).

## Intended Use

- **Primary use:** Predict whether a person living in the United States makes over 50K/year using 1994 Census income data.
- **Users:** Data scientists/analysts exploring this dataset for educational or prototyping purposes.

## Out-of-Scope Use

- Any real-world decision making about individuals (e.g., lending, hiring, or benefits decisions).
- Production deployment without updated, representative data and a thorough fairness review.

## Training Data

- **Source:** `data/census.csv` (UCI Census Income dataset).
- **Collection window:** 1994 US Census (as described by the dataset source).
- **Size and split:** 80% of rows used for training, 20% for evaluation (`train_test_split(test_size=0.20, random_state=42)`).
- **Label column:** `salary`.
- **Input features:**
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
  - Continuous: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Preprocessing:**
  - Whitespace stripped from column names.
  - One-hot encoding for categorical features (`OneHotEncoder(handle_unknown="ignore")`).
  - Label binarization for `salary` (`LabelBinarizer`).
  - No explicit scaling or normalization of continuous features.

## Evaluation Data

- **Source:** Same dataset as training (`data/census.csv`).
- **Split:** 20% holdout test set from `train_test_split` with `random_state=42`.
- **Preprocessing:** Identical to training (one-hot encoding and label binarization with the training-fitted encoders).

## Metrics

- **Primary metrics:** Precision, recall, and F1 (fbeta with beta=1).
- **Computation:** `ml/model.py::compute_model_metrics` on the test set in `train_model.py`.
- **Slice metrics:** `model_metrics_slices.py` computes precision/recall/F1 by `education` category and writes the results to `screenshots/slice_output.json`.

## Ethical Considerations

- The model uses sensitive demographic attributes (e.g., `race`, `sex`, `native-country`), which can encode societal biases.
- The data is from the 1994 Census, making it outdated and likely unrepresentative of current populations.
- Slice analysis by `education` is available (`screenshots/slice_output.json`), but additional fairness analysis is recommended before any deployment.

## Caveats and Recommendations

- **Data drift risk:** The model is trained on historical census data and should not be assumed valid for modern populations.
- **Bias risk:** Demographic attributes are used as inputs; results should be interpreted with caution and bias audits.
- **Recommendation:** Limit usage to educational or exploratory purposes unless re-trained with current data and validated for fairness.

## Training Procedure

- **Algorithm:** Random Forest classifier.
- **Training workflow:**
  1. Load `data/census.csv` and strip column whitespace.
  2. Split data into train/test (80/20, `random_state=42`).
  3. Preprocess data via `ml/data.py::process_data`.
  4. Fit `GridSearchCV` (5-fold) to select best hyperparameters.
  5. Evaluate on the test set and save model + metrics.
- **Artifacts saved:**
  - `model/model.pkl`
  - `model/encoder.pkl`
  - `model/lb.pkl`
  - `model/precision.json`
  - `model/recall.json`
  - `model/fbeta.json`

## Reproducibility

- **Environment:** Python with dependencies listed in `requirements.txt`.
- **Key dependencies:** `numpy`, `scikit-learn`, `fastapi`, `uvicorn`.
- **Retraining steps:**
  1. Ensure `data/census.csv` is present.
  2. Install dependencies: `pip install -r requirements.txt`.
  3. Run training: `python train_model.py`.
  4. (Optional) Generate slice metrics: `python model_metrics_slices.py`.
