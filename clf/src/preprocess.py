import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def load_data():
    df = pd.read_excel('../data/data.xlsx', sheet_name='E Comm')
    desc = pd.read_excel('../data/data.xlsx', sheet_name='Data Dict', header=1, usecols=[1,2,3]).drop(columns="Data")
    return df, desc

def clean_data(df):
    df.drop(columns="CustomerID", inplace=True)

def impute_data(df):
    numerics = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerics] = SimpleImputer(strategy='mean').fit_transform(df[numerics])

    categoricals = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categoricals, drop_first=True)

    rf_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=250), max_iter=10)
    df = pd.DataFrame(rf_imputer.fit_transform(df), columns=df.columns)

def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=250, stratify=y)
    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train):
    smote = SMOTE(random_state=250)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled