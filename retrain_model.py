# retrain_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5]  # SepalLength, SepalWidth, PetalLength, PetalWidth
y = df["Species"]

# Split and train
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "savedmodel.sav")
print("✅ Model retrained and saved.")
