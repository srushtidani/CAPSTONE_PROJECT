import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = 'Fraud_Analysis_Dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Preprocessing
# Encoding categorical features (e.g., 'type')
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Selecting features and target
X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)  # Features
y = df['isFraud']  # Target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier(
    criterion='gini',       # Splitting criterion: 'gini' for Gini impurity, 'entropy' for Information Gain
    max_depth=10,           # Maximum depth of the tree
    min_samples_split=10,   # Minimum number of samples required to split an internal node
    min_samples_leaf=5,     # Minimum number of samples required to be at a leaf node
    max_features=None,      # Number of features to consider for the best split (None means all)
    random_state=42         # Seed for reproducibility
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model using joblib
model_filename = 'decision_tree_fraud_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")
