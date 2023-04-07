import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    # print(dataframe_train_typed.isnull().any())
    # print(dataframe_test_typed.isnull().any())

    # Generate some new fields by extracting data from the existing fields
    generate_new_fields(dataframe_train_typed)
    generate_new_fields(dataframe_test_typed)

    # Combine the training and test data
    all_data = pd.concat([dataframe_train_typed, dataframe_test_typed], sort=False)

    # List out features we care about
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "HasFamily", "FamilySize", "Title", "IsChild", "IsElderly"]

    # One-hot encode the 'Title' feature
    all_data_features = get_features(all_data, features)

    # Split the combined data back into training and test data
    train_data = all_data_features.iloc[:len(dataframe_train_typed)]
    test_data = all_data_features.iloc[len(dataframe_train_typed):]

    model = LogisticRegression()
    print(all_data_features.columns)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    model.fit(train_data, dataframe_train_typed["Survived"])
    predictions = model.predict(test_data)
    int_predictions = predictions.astype(np.int64)

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


def fillna(dataframe: pd.DataFrame) -> None:
    # Set any null values in the age column to the median age
    dataframe["Age"].fillna(dataframe["Age"].median(), inplace=True)

    # Set any null values in the embarked and cabin columns to the most common values
    dataframe["Embarked"].fillna(dataframe["Embarked"].mode()[0], inplace=True)
    dataframe["Cabin"].fillna(dataframe["Cabin"].mode()[0], inplace=True)
    dataframe["Fare"].fillna(dataframe["Fare"].mean(), inplace=True)

    return


def generate_new_fields(dataframe: pd.DataFrame) -> None:
    dataframe["HasFamily"] = dataframe["SibSp"] + dataframe["Parch"] > 1
    dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    dataframe["Title"] = dataframe["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    dataframe["IsChild"] = dataframe["Age"].apply(lambda x: 1 if x < 18 else 0)
    dataframe["IsElderly"] = dataframe["Age"].apply(lambda x: 1 if x > 60 else 0)
    return


def get_features(dataframe: pd.DataFrame, features: list[str]):
    return pd.get_dummies(dataframe[features])


if __name__ == "__main__":
    main()
