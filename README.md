## 📌 Project Overview
This project builds a machine learning model to classify text into:
- Hate Speech
- Offensive Language
- No Hate or Offensive Language

It demonstrates an end-to-end workflow: data preprocessing, feature extraction, model training, evaluation, and prediction.

---

## ⚙️ Approach
1. **Data Preprocessing**
   - Lowercasing, removing punctuation, URLs, HTML tags, and numbers
   - Stopword removal and stemming using NLTK

2. **Feature Engineering**
   - Bag-of-Words representation with `CountVectorizer`

3. **Model**
   - Decision Tree Classifier

4. **Evaluation**
   - Accuracy score
   - Confusion matrix visualization

---

## 🛠️ Tools & Libraries
- Python
- NumPy, pandas
- scikit-learn
- NLTK
- matplotlib, seaborn
- GitHub for version control and project sharing

---

## 📊 Results
- Model Accuracy: ~86%  
- Confusion Matrix:

![Confusion Matrix](results/confusion_matrix.png)

---

## 🚀 How to Run
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/hate-speech-detection.git
cd hate-speech-detection
pip install -r requirements.txt
