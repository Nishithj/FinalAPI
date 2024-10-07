# Import necessary libraries
import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from collections import Counter
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all domains on /api route

# Define allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

# Load pre-trained machine learning models
model1 = load_model("C:\Users\HP\OneDrive\Desktop\Final_API\models\my_model.h5")
model2 = load_model("C:\Users\HP\OneDrive\Desktop\Final_API\models\lstm1.h5")
model3 =  load_model("C:\\Users\\HP\\OneDrive\\Desktop\\Final_API\\models\\cnnlstm.h5")
# Function to check if the file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function for audio preprocessing and feature extraction
def preprocess_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Preprocessing steps: normalize audio
    y = librosa.util.normalize(y)

    # Feature extraction: MFCC, NAQ, F0
    f0 = np.mean(librosa.piptrack(y=y, sr=sr)[0])
    naq = np.mean(librosa.feature.rms(y=y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)

    # Stack features
    features = np.concatenate(([f0,naq],np.mean(mfccs, axis=1)))

    return features

# API endpoint for audio classification
@app.route('/api', methods=['GET', 'POST'])
def classify_audio():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Create the uploads directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Preprocess audio and extract features
            features = preprocess_audio(file_path)

            # Make predictions using the pre-trained models
            prediction1 = int(np.argmax(model1.predict(np.expand_dims(features, axis=0))))
            prediction2 = int(np.argmax(model2.predict(np.expand_dims(features, axis=0))))
            prediction3 = int(np.argmax(model3.predict(np.expand_dims(features, axis=0))))

            # Perform voting
            predictions = [prediction1, prediction2, prediction3]
            count = Counter(predictions)
            majority_vote = count.most_common(1)[0][0]

            # Return the majority vote
            return jsonify({'majority_vote': majority_vote})

        else:
            return jsonify({'error': 'File format not supported'})
    else:
        return jsonify({'message': 'Send a POST request with an audio file'})

if __name__ == '__main__':
    app.run(debug=True)
