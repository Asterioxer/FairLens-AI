# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path="data/adult.csv"):
    # Define column names (UCI Adult dataset doesn't include headers)
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(file_path, names=columns, na_values=" ?", skipinitialspace=True)
    df = df.dropna()  # Remove missing values

    # Define features, sensitive attributes, and target
    features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'hours-per-week']
    sensitive_attributes = ['sex', 'race']
    target = 'income'

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df[features]
    y = df[target]
    sensitive_data = df[sensitive_attributes]

    return X, y, sensitive_data

# Test in Colab
X, y, sensitive_data = load_and_preprocess_data()
print("Data loaded:", X.shape, y.shape, sensitive_data.shape)
