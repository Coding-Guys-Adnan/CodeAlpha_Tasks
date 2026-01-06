# Iris Classification using Custom Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load your dataset
# Replace path with your actual file path
data = pd.read_csv("iris.csv")

# 2. Separate features and target
X = data.drop("Species", axis=1)
y = data["Species"]

# 3. Encode target labels if they are text
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create and train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=encoder.classes_))
