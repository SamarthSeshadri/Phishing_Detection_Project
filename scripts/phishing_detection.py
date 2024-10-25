import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from flask import Flask, request, jsonify

# Download NLTK stop words and WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Define file paths
phishing_email_path = 'datasets/phishing_email.csv'
nigerian_fraud_path = 'datasets/Nigerian_Fraud.csv'
spam_assassin_path = 'datasets/SpamAssasin.csv'
ceas_08_path = 'datasets/CEAS_08.csv'
enron_path = 'datasets/Enron.csv'
ling_path = 'datasets/Ling.csv'
nazario_path = 'datasets/Nazario.csv'

# Load datasets
phishing_email = pd.read_csv(phishing_email_path)
nigerian_fraud = pd.read_csv(nigerian_fraud_path)
spam_assassin = pd.read_csv(spam_assassin_path)
ceas_08 = pd.read_csv(ceas_08_path)
enron = pd.read_csv(enron_path)
ling = pd.read_csv(ling_path)
nazario = pd.read_csv(nazario_path)

# Initialize an empty DataFrame for the combined data
combined_data = pd.DataFrame(columns=['text', 'label'])

# Combine datasets into a single DataFrame
datasets = [phishing_email, nigerian_fraud, spam_assassin, ceas_08, enron, ling, nazario]

for dataset in datasets:
    if 'text_combined' in dataset.columns:
        dataset_cleaned = dataset.rename(columns={'text_combined': 'text'})[['text', 'label']]
    elif 'body' in dataset.columns:
        dataset_cleaned = dataset.rename(columns={'body': 'text'})[['text', 'label']]
    elif 'content' in dataset.columns:
        dataset_cleaned = dataset.rename(columns={'content': 'text'})[['text', 'label']]
    combined_data = pd.concat([combined_data, dataset_cleaned], ignore_index=True)

# Clean the text data
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english'))
    return text

combined_data['text'] = combined_data['text'].astype(str).fillna('').apply(clean_text)

# Ensure labels are integers
combined_data['label'] = combined_data['label'].astype(int)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(combined_data['text']).toarray()
y = combined_data['label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm shapes of the training and testing sets
print("Shapes of the datasets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning for the best model (example with Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Save the model
joblib.dump(grid_search.best_estimator_, 'models/phishing_detection_model.pkl')

# Load the model
model = joblib.load('models/phishing_detection_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json['text']
    email_text = clean_text(email_text)
    features = tfidf.transform([email_text]).toarray()
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
