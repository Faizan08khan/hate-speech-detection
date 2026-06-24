 -----------------------------
# Load dataset
# -----------------------------
import pandas as pd
dataset = pd.read_csv("X_hate.csv")

dataset["labels"] = dataset["class"].map({
    0: "Hate Speech",
    1: "Offensive language",
    2: "No hate or offensive language"
})
data = dataset[["tweet", "labels"]]

# -----------------------------
# Preprocessing
# -----------------------------
import re, string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = [word for word in text.split() if word not in stopwords_set]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

data.loc[:, "tweet"] = data["tweet"].apply(clean_data)

# -----------------------------
# Train/Test Split
# -----------------------------
import numpy as np
from sklearn.model_selection import train_test_split

X_text = data["tweet"].astype(str)
y = np.array(data["labels"])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train_text)
X_test = cv.transform(X_test_text)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
plt.show()

print("Model Accuracy:", acc)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_text(text):
    cleaned = clean_data(text)
    features = cv.transform([cleaned])
    return dt.predict(features)[0]

# Example usage
print(predict_text("I hate you"))
print(predict_text("You are awesome"))
