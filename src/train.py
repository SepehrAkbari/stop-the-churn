from preprocess import *
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df, desc = load_data()
clean_data(df)
impute_data(df)
X_train, X_test, y_train, y_test = split_data(df)
X_train_resampled, y_train_resampled = balance_data(X_train, y_train)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    model = XGBClassifier(random_state=250)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_pred_proba

if __name__ == "__main__":
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train_resampled, X_test)
    model, y_pred, y_pred_proba = train_model(X_train_scaled, y_train_resampled)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)