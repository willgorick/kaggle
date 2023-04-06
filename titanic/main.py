import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import numpy as np

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission.csv"
PATH = Path("data/")


def main() -> None:
    # Read in both the training and test files
    dataframe_train = read_file(PATH, TRAIN_FILE)
    dataframe_test = read_file(PATH, TEST_FILE)

    # Add the proper types to our dataframe so we can store the data more efficiently
    dataframe_train_typed = type_dataframe(dataframe_train)
    dataframe_test_typed = type_dataframe(dataframe_test)

    # Shows that Age, Cabin, and Embarked all have null values
    # print(dataframe_train_typed.isnull().any())

    # Fill in null values
    fillna(dataframe_train_typed)
    fillna(dataframe_test_typed)

    # Shows that these fields no longer have any null values
    print(dataframe_train_typed.isnull().any())
    print(dataframe_test_typed.isnull().any())

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    training_features = get_features(dataframe_train_typed, features)
    test_features = get_features(dataframe_test_typed, features)
    
    model = LogisticRegression()
    model.fit(training_features, dataframe_train_typed["Survived"])
    
    predictions = model.predict(test_features)
    int_predictions = predictions.astype(np.int64)

    print(int_predictions)

    output = pd.DataFrame({'PassengerId': dataframe_test_typed.PassengerId, 'Survived': int_predictions})
    write_file(PATH, SUBMISSION_FILE, output)
    return

def read_file(path: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(path / filename)

def write_file(path: Path, filename: str, output: pd.DataFrame):
    output.to_csv(path / filename, index=False)

def type_dataframe(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    mapping_type_conversions = {
        "PassengerId": "int64",
        "Pclass": "category",
        "Name": "string",
        "Sex": "category",
        "Age": "float64",
        "SibSp":     "int64",
        "Parch":     "int64",
        "Ticket":    "string",
        "Fare":      "float64",
        "Cabin":     "string",
        "Embarked":  "category"
    }

    typed_dataframe = raw_dataframe.astype(mapping_type_conversions)
    if "Survived" in typed_dataframe:
        typed_dataframe["Survived"] = typed_dataframe["Survived"].astype("bool")
    return typed_dataframe

def fillna(dataframe: pd.DataFrame):
    # Set any null values in the age column to the median age
    dataframe["Age"].fillna(dataframe["Age"].median(), inplace=True)

    # Set any null values in the embarked and cabin columns to the most common values
    dataframe["Embarked"].fillna(dataframe["Embarked"].mode()[0], inplace=True)
    dataframe["Cabin"].fillna(dataframe["Cabin"].mode()[0], inplace=True)
    dataframe["Fare"].fillna(dataframe["Fare"].mean(), inplace=True)

    return

def get_features(dataframe: pd.DataFrame, features: list[str]):
    return pd.get_dummies(dataframe[features])

if __name__ == "__main__":
    main()
