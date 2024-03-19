# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load your historical customer data (assuming CSV format)
df = pd.read_csv('Churn_Modelling.csv')

# Select features (excluding 'RowNumber', 'CustomerId', 'Surname', and 'Exited' columns)
features = df.iloc[:, 3:-1]

# Encode categorical features (Gender and Geography)
features = pd.get_dummies(features)

# Target variable: 'Exited' (1 for churned, 0 for not churned)
target = df['Exited']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model with training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
