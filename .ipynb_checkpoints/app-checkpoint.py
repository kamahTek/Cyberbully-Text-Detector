from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# create flask app
app = Flask(__name__, template_folder='Web-App/templates', static_folder='Web-App/static')

# Load the entire model
model = tf.keras.models.load_model('best_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define the route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get("text")

        # Preprocess the text (tokenize and pad as per the training steps)
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=500)
        
        # Process and make predictions
        # Predicts a single label (0, 1, or 2)
        prediction = model.predict(padded)[0]  
        # Get the class with the highest probability
        prediction_class = prediction.argmax()
        
        # Map class numbers to labels
        class_labels = {0: 'hate-speech', 1: 'offensive-language', 2: 'neither'}
        prediction_label = class_labels.get(prediction_class, "Unknown")

        return render_template("index.html", text_prediction = prediction_label)

if __name__ == '__main__':
    app.run(debug=True)