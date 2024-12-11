import json
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
import random

app = Flask(__name__)
api = Api(app)

MODEL_FILENAME = 'chatbot_model.joblib'

with open('data/chatbot_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

intents = []
patterns = []
responses = {}
context_set = {}

for intent in data:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    context_set[intent['tag']] = intent.get('context_set', None)

le = LabelEncoder()
intents_encoded = le.fit_transform(intents)

X_train, X_test, y_train, y_test = train_test_split(patterns, intents_encoded, test_size=0.3, random_state=42)

def load_or_train_model():
    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        print("Model loaded successfully.")
    else:
        print("Model not found, training a new model.")
        model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
        model.fit(X_train, y_train)
        
        joblib.dump(model, MODEL_FILENAME)
        print("Model trained and saved.")
    
    return model

model = load_or_train_model()

accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_intent(text):
    predicted_class = model.predict([text])[0]
    intent = le.inverse_transform([predicted_class])[0]
    return intent

def get_best_matching_pattern(user_input):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)
    user_input_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_input_vector, X)
    best_match_index = np.argmax(similarity_scores)
    return best_match_index, similarity_scores[0][best_match_index]

def get_bot_response(user_input, user_context=None):
    best_match_index, similarity_score = get_best_matching_pattern(user_input)
    
    intent = intents[best_match_index]
    
    if similarity_score < 0.2:
        return "Maaf, AskResikel hanya dapat menjawab tentang pengelolaan sampah atau dokumentasi terkait aplikasi Resikel. Silakan ajukan pertanyaan seputar topik tersebut."

    context_required = context_set.get(intent)
    if context_required and user_context != context_required:
        return "Sorry, I can't respond to that right now. Please continue the previous conversation."

    response = random.choice(responses[intent])
    return response

hardcode_api_key = '499c18c6-9f57-45f8-b6eb-ba2c8275e274'

class ChatbotAPI(Resource):
    def post(self):
        api_key = request.headers.get('API-Key')
        if api_key != hardcode_api_key:
            return jsonify({'error': 'Invalid API key. Please check your API key and try again.'})

        user_input = request.json.get('message')
        user_context = request.json.get('context')
        if not user_input:
            return jsonify({'error': 'No message provided'})

        response = get_bot_response(user_input, user_context)
        return jsonify({'response': response})

api.add_resource(ChatbotAPI, '/chat')

def show_confusion_matrix():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # show_confusion_matrix()
    app.run(debug=True)
