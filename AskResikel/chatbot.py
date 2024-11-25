import json
import random
import joblib
from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
api = Api(app)

MODEL_FILENAME = 'chatbot_model.joblib'

with open('data/chatbot_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

intents = []
patterns = []
responses = {}

for intent in data:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

le = LabelEncoder()
intents_encoded = le.fit_transform(intents)

X_train, X_test, y_train, y_test = train_test_split(patterns, intents_encoded, test_size=0.2, random_state=42)

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

def get_bot_response(user_input):
    intent = predict_intent(user_input)
    
    if intent in responses:
        response = random.choice(responses[intent])
    else:
        response = "Sorry, I didn't understand that. Could you please rephrase?"
    return response

class ChatbotAPI(Resource):
    def post(self):
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No message provided'})

        response = get_bot_response(user_input)
        return jsonify({'response': response})

api.add_resource(ChatbotAPI, '/chat')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
