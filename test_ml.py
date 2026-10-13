import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """Full census dataset."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split(data):
    """Stratified 80/20 train/test split, matching train_model.py."""
    return train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )


@pytest.fixture(scope="module")
def processed(split):
    """Processed train/test arrays plus the fitted encoder and label binarizer."""
    train, test = split
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb


@pytest.fixture(scope="module")
def model(processed):
    """Model trained on the processed training data."""
    X_train, y_train, *_ = processed
    return train_model(X_train, y_train)


def test_split_size_and_types(data, split):
    """
    The train/test split covers every row with an 80/20 ratio, keeps all
    columns, and preserves the label's class balance in both sets.
    """
    train, test = split
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)
    assert list(train.columns) == list(data.columns)
    assert list(test.columns) == list(data.columns)

    overall = (data["salary"] == ">50K").mean()
    assert (train["salary"] == ">50K").mean() == pytest.approx(overall, abs=0.01)
    assert (test["salary"] == ">50K").mean() == pytest.approx(overall, abs=0.01)


def test_process_data_output(split, processed):
    """
    process_data returns numeric arrays with one row per input row, the same
    number of features for train and test, and binary 0/1 labels.
    """
    train, test = split
    X_train, y_train, X_test, y_test, _, _ = processed

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0] == len(train)
    assert X_test.shape[0] == y_test.shape[0] == len(test)
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)) == {0, 1}


def test_train_model_algorithm(model):
    """
    train_model returns a fitted RandomForestClassifier.
    """
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")  # only present after fit()


def test_inference_output(model, processed):
    """
    inference returns a NumPy array with one binary prediction per input row.
    """
    _, _, X_test, *_ = processed
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X_test.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    """
    compute_model_metrics returns the expected precision, recall, and F1 on
    a hand-checked example: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3.
    """
    y = np.array([1, 1, 1, 0, 0, 0])
    preds = np.array([1, 1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_save_and_load_model(model, processed, tmp_path):
    """
    A model saved with save_model and reloaded with load_model produces the
    same predictions as the original.
    """
    _, _, X_test, *_ = processed
    path = tmp_path / "model.pkl"
    save_model(model, path)
    loaded = load_model(path)

    assert isinstance(loaded, RandomForestClassifier)
    np.testing.assert_array_equal(inference(loaded, X_test), inference(model, X_test))


def test_apply_label():
    """
    apply_label maps binary predictions back to the original salary strings.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
