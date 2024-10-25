import joblib
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load the saved model
model = joblib.load(os.path.join(os.path.dirname(__file__), 'phishing_detection_model.pkl'))

# Load the TF-IDF vectorizer
tfidf = joblib.load(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'))

# Create Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')
socketio = SocketIO(app)

# Function to clean and preprocess the input text
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english'))
    return text

def highlight_suspicious_elements(text):
    # Highlight suspicious elements such as links and common phishing keywords
    suspicious_keywords = ['win', 'prize', 'urgent', 'verify', 'click', 'password', 'account']
    for keyword in suspicious_keywords:
        text = re.sub(f'\\b{keyword}\\b', f'<span class="highlight">{keyword}</span>', text, flags=re.IGNORECASE)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json or 'text' not in request.json:
            return jsonify({'error': 'Invalid input'}), 400

        email_text = request.json['text']
        print(f'Input text: {email_text}')
        email_text = clean_text(email_text)
        print(f'Cleaned text: {email_text}')
        features = tfidf.transform([email_text]).toarray()
        prediction = model.predict(features)
        print(f'Prediction: {prediction}')
        confidence = model.predict_proba(features).max()  # Get the highest probability
        print(f'Confidence: {confidence}')

        confidence_threshold = 0.95
        if confidence < confidence_threshold:
            prediction = 0  # Not a phishing attempt

        highlighted_text = highlight_suspicious_elements(email_text)
        print(f'Highlighted text: {highlighted_text}')
        # Emit a real-time alert
        socketio.emit('alert', {'prediction': int(prediction[0]), 'confidence': confidence, 'highlighted_text': highlighted_text})
        return jsonify({'prediction': int(prediction[0]), 'confidence': confidence, 'highlighted_text': highlighted_text})

    except ValueError as ve:
        return jsonify({'error': 'Value error', 'message': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)
