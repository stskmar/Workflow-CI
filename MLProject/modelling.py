import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Enable autologging
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv("preprocessing/bank_clean.csv")
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
