import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load and clean dataset
df = pd.read_csv('credit.csv')  # Make sure 'credit.csv' is in the same folder
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
print("Columns:", df.columns)

# Confirm actual column name for target
target_column = 'creditability'  # Change if necessary based on df.columns output

# Encode categorical data
df = df.apply(LabelEncoder().fit_transform)

# Features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
