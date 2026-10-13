# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by Andrew Peebles in October 2026 as part of the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project. It is a binary classifier that predicts whether a person's annual income is above $50K (`>50K`) or at or below $50K (`<=50K`) based on census attributes.

The model is a scikit-learn 1.5.1 `RandomForestClassifier` with 100 trees, a maximum depth of 15, a minimum of 2 samples per leaf, and `random_state=42` for reproducibility. The depth and leaf limits were chosen because they shrink the serialized model from about 74 MB to about 9.5 MB while lowering the test F1 score by only about 0.005. Categorical features are one-hot encoded with a `OneHotEncoder` that ignores unseen categories, and the label is binarized with a `LabelBinarizer`, so `>50K` is mapped to 1 and `<=50K` is mapped to 0. Continuous features are passed through without scaling, which is acceptable for a tree-based model. The trained model and encoder are saved as pickle files in the `model/` directory and are served through a FastAPI application.

## Intended Use

The model is intended for educational purposes, specifically to demonstrate how to build, evaluate, test, and deploy a machine learning pipeline with continuous integration and a REST API. It can be used to explore how demographic and employment attributes relate to income in the 1994 census sample and to study how a model's performance varies across subgroups.

The model is not intended for making real decisions about individuals, such as decisions about hiring, lending, credit, insurance, housing, or eligibility for public benefits. The data is decades old and the model shows uneven performance across demographic groups, so using it in any of those settings could cause harm.

## Training Data

The model was trained on the UCI Adult ("Census Income") dataset, which was extracted by Barry Becker from the 1994 U.S. Census database and is available at https://archive.ics.uci.edu/dataset/2/adult. The version in `data/census.csv` contains 32,561 rows and 15 columns. There are six continuous features (`age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, and `hours-per-week`), eight categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`), and the label `salary`.

The label is imbalanced, with about 75.9% of rows labeled `<=50K` and about 24.1% labeled `>50K`. The dataset has no empty cells, but missing values are recorded as `?` in `workclass` (1,836 rows), `occupation` (1,843 rows), and `native-country` (583 rows). These `?` values were kept and treated as their own category. The data was split into 80% training data (26,048 rows) and 20% evaluation data (6,513 rows) using `train_test_split` with `random_state=42`, stratified on the label so that both sets keep the same class balance. The encoder and label binarizer were fit on the training set only.

## Evaluation Data

The evaluation data is the 20% hold-out split described above, which contains 6,513 rows. It was processed with the encoder and label binarizer that were fit on the training data, so no information from the evaluation set influenced training. In addition to scoring the full evaluation set, the model was scored on every slice formed by fixing a single value of each categorical feature, which produced 101 slices. The full per-slice results are in `slice_output.txt`.

## Metrics

The model was evaluated with precision, recall, and F1 score, where the positive class is `>50K`. Precision measures how many of the people predicted to earn more than $50K actually do, recall measures how many of the people who earn more than $50K the model correctly finds, and F1 is the harmonic mean of the two. These metrics were chosen over accuracy because the label is imbalanced, and a model that always predicted `<=50K` would still be about 76% accurate.

On the full evaluation set, the model achieved the following performance:

| Metric | Value |
|---|---|
| Precision | 0.8009 |
| Recall | 0.5874 |
| F1 | 0.6777 |

The model is fairly precise, but it misses about 41% of people who earn more than $50K. The slice analysis shows that performance differs across groups:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| sex: Male | 4,355 | 0.7897 | 0.6017 | 0.6830 |
| sex: Female | 2,158 | 0.8803 | 0.5102 | 0.6460 |
| race: White | 5,533 | 0.8069 | 0.5929 | 0.6836 |
| race: Black | 662 | 0.8000 | 0.4731 | 0.5946 |
| race: Asian-Pac-Islander | 200 | 0.6667 | 0.6538 | 0.6602 |
| education: Masters | 318 | 0.8426 | 0.9121 | 0.8760 |
| education: Bachelors | 1,096 | 0.7470 | 0.8650 | 0.8017 |
| education: HS-grad | 2,120 | 0.9506 | 0.2271 | 0.3667 |
| relationship: Own-child | 1,032 | 0.6667 | 0.1333 | 0.2222 |

The model performs best for people with advanced degrees and worst for people with a high school education or less, for people who are not married, and for people in lower-wage occupations such as `Other-service`. Recall is lower for women than for men and lower for Black individuals than for White individuals, which means that high earners in those groups are more likely to be missed.

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, marital status, and native country, and the model uses all of them as features. Because the data reflects income inequality that existed in 1994, the model can learn and repeat those patterns. The slice results confirm that the model's errors are not evenly distributed, since recall is about 0.09 lower for women than for men and about 0.12 lower for Black individuals than for White individuals.

Several subgroups are very small in the evaluation set. For example, there are 73 rows for `race: Amer-Indian-Eskimo`, 45 rows for `race: Other`, and a single row each for `workclass: Never-worked` and `workclass: Without-pay`. Metrics on slices this small are unreliable and should not be read as evidence that the model is fair or unfair to those groups. In addition, the `fnlgt` feature is a census sampling weight rather than a property of the person, so its use as a predictor has no clear real-world meaning.

## Caveats and Recommendations

The data is from 1994, and the $50K threshold has not been adjusted for inflation or changes in the labor market, so the model should not be expected to generalize to present-day populations. The precision values in `slice_output.txt` should be read carefully, because metrics are computed with `zero_division=1`. When the model makes no positive predictions on a slice, precision is reported as 1.0 even though recall may be 0.0. For example, the `education: 7th-8th` slice shows a precision of 1.0000, a recall of 0.0000, and an F1 of 0.0000.

To improve the model, future work could tune hyperparameters with cross-validation, adjust the decision threshold or use class weights to improve recall on the minority class, and evaluate the effect of removing sensitive attributes and `fnlgt`. Fairness metrics such as equal opportunity difference should also be tracked alongside the slice metrics. Anyone retraining the model should rerun `train_model.py` and review `slice_output.txt` before deploying a new version.
