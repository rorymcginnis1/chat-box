import random
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load intents from JSON file
intents = json.loads(open('response.json').read())

# Extract questions, responses, and labels from intents
questions = []
responses = []
labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        questions.append(pattern)
        responses.append(intent['responses'])  
        labels.append(intent['tag'])


tokenizer = Tokenizer()

tokenizer.fit_on_texts(questions)
questions_seq = tokenizer.texts_to_sequences(questions)

# Define the model architecture
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
    LSTM(16),
    Dense(len(intents['intents']), activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


padded_sequences = pad_sequences(questions_seq, maxlen=10, padding='post', truncating='post')
questions_seq_np = np.array(padded_sequences)


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


model.fit(questions_seq_np, encoded_labels, epochs=500, batch_size=4, verbose=1)



# Function to get a random response from an intent
def get_random_response(intent):
    responses = intent['responses']
    return random.choice(responses)



while True:
    user_input = input("You: ") 


    input_seq = tokenizer.texts_to_sequences([user_input])


    predicted_probs = model.predict(input_seq)[0]


    max_prob_index = np.argmax(predicted_probs)
    predicted_intent_tag = label_encoder.inverse_transform([max_prob_index])[0]


    predicted_intent = next(intent for intent in intents['intents'] if intent['tag'] == predicted_intent_tag)


    response = get_random_response(predicted_intent)


    print("ChatBot:", response)
