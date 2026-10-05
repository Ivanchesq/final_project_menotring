import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def process_data(file_path, is_train=True, age_medians=None, fare_median=None, embarked_mode=None, age_median=None):
    df = pd.read_csv(file_path)

    df['Age_Is_Null'] = df['Age'].isnull().astype(int)
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    df.drop('Cabin', axis=1, inplace=True, errors='ignore')

    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

    if is_train:
        embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df.drop('Ticket', axis=1, inplace=True, errors='ignore')

    if is_train:
        age_medians = df.groupby('Pclass')['Age'].median().to_dict()
        fare_median = df['Fare'].median()
        age_median = df['Age'].median()

    for pclass, median_val in age_medians.items():
        mask = (df['Age'].isna()) & (df['Pclass'] == pclass)
        df.loc[mask, 'Age'] = median_val
    df['Age'] = df['Age'].fillna(age_median)

    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Fare'] = np.log1p(df['Fare'])

    pclass_type = CategoricalDtype(categories=[1, 2, 3], ordered=True)
    df['Pclass'] = df['Pclass'].astype(pclass_type)
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)

    if is_train:
        return df, age_medians, fare_median, embarked_mode, age_median
    else:
        return df