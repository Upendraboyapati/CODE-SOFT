#importing required modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the SMS spam dataset (assuming it's in a file named 'spam.csv')
df = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocess the text (remove stopwords, punctuation, etc.)
# You can add more preprocessing steps as needed

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_tfidf = vectorizer.fit_transform(df['v2'])  # Assuming 'v2' contains SMS text

# Map labels to binary values (spam: 1, ham: 0)
df['label'] = df['v1'].map({'spam': 1, 'ham': 0})
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate model performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
