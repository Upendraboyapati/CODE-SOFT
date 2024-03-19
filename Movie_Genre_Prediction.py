# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load datasets
trainset = pd.read_csv("train_data.csv",delimiter=":::",engine='python',header=None,names=["id","moviename","genre","plot"])
testset = pd.read_csv("test_data.csv",delimiter=":::",engine='python',header=None,names=["id","moviename","plot"])

# Extract features using TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(trainset['plot'])
y_train = trainset['genre']

# Scale your data
scaler = StandardScaler(with_mean=False)  # Use with_mean=False to avoid a sparse matrix issue
X_train_scaled = scaler.fit_transform(X_train)

# Train a logistic regression classifier with increased number of iterations
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_scaled, y_train)

# Transform test set plots into TF-IDF vectors 
X_test = vectorizer.transform(testset['plot'])

# Scale the test data using the same scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)

# Predict genres for the test set plots 
predicted_genres = classifier.predict(X_test_scaled)

# Print predicted genres - you can also save these to a file or dataframe if needed.
print(predicted_genres)
