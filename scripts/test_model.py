import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and TF-IDF vectorizer
model = joblib.load('phishing_detection_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Function to clean and preprocess the input text
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english'))
    return text

# Test inputs
test_inputs = [
    "Congratulations! You have won a $1000 gift card. Click here to claim your prize.",
    "Your account has been compromised. Click here to reset your password.",
    "Dear user, please update your payment information immediately.",
    "Hi, I wanted to check in and see if you're available for a meeting next week.",
    "Your package has been shipped. Track your order here.",
    "URGENT: Your bank account is at risk. Verify your identity now.",
    "Please find attached the meeting agenda for next week."
]

for text in test_inputs:
    cleaned_text = clean_text(text)
    features = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(features)
    confidence = model.predict_proba(features).max()
    print(f'Text: {text}')
    print(f'Prediction: {prediction[0]}, Confidence: {confidence}')
    print('---')
