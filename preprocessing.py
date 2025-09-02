import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Assume last column = target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)
