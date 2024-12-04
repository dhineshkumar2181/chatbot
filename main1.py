from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Download necessary NLTK data (you can remove this after first-time setup)
nltk.download('punkt')
nltk.download('wordnet')

# Function to clean up and tokenize sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    try:
        res = model.predict(np.array([bow]))[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return []
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if len(results) == 0:
        return []

    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get a response
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that. Could you please rephrase?"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-response', methods=['POST'])
def get_bot_response():
    user_message = request.json.get('message')
    
    if not user_message or len(user_message.strip()) == 0:
        return jsonify({'response': "Please enter a message. I'm here to help!"})
    
    # Get intent predictions
    try:
        ints = predict_class(user_message)
        if not ints:
            bot_response = "Sorry, I couldn't understand that. Can you try asking something else?"
        else:
            bot_response = get_response(ints, intents)
    except Exception as e:
        print(f"Error while processing the user message: {e}")
        bot_response = "Oops! Something went wrong on my side. Please try again later."

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
