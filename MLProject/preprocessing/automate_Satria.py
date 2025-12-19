import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    input_path: str,
    output_path: str
):
    # Load data
    df = pd.read_csv(input_path, sep=';')

    # Encode target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Drop duplicates
    df = df.drop_duplicates()

    # One-hot encoding categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Standardization
    numerical_cols = [
        'age', 'balance', 'day',
        'duration', 'campaign', 'pdays', 'previous'
    ]

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Save processed data
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    preprocess_data(
        input_path="Membangun_model/preprocessing/bank.csv",
        output_path="Membangun_model/preprocessing/bank_clean.csv"
    )
