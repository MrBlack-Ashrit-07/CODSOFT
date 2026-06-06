# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "Label", "v2": "Messages"})
print(df.head())
print(df.info())
X = df["Messages"]
y = df["Label"]


# Split Test/Train Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# Tfidf Vectorizer
vectorizer = TfidfVectorizer()
X_train_Tfidf = vectorizer.fit_transform(X_train)
X_test_Tfidf = vectorizer.transform(X_test)

# Model Creation 
model = MultinomialNB()

#Model Training
model.fit(X_train_Tfidf, y_train)

#Prediction & Accuracy Score
prediction = model.predict(X_test_Tfidf)
accuracy = accuracy_score(y_test, prediction)
print(accuracy)

