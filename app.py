from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import pandas as pd

# Init Flask app
app = Flask(__name__)

# Path to model
model_path = 'model/model.keras'

# Load the model
MODEL = load_model(model_path)

# Set up params
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MAX_LENGTH = 180000

# Load Song Data
try:
    song_data = pd.read_csv("D:\\Neuron\\final\\songs.csv")
    print("Song data loaded successfully.")
except FileNotFoundError:
    print("Error: song.csv not found. Song suggestions will be disabled.")
    song_data = None
except Exception as e:
    print(f"Error loading song.csv: {e}. Song suggestions will be disabled.")
    song_data = None

# Từ điển mã hóa cảm xúc
emotion_dic = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'disgust': 5
}

reverse_emotion_dic = {v: k for k, v in emotion_dic.items()}


def preprocess_audio(path):
    try:
        # Tạo đối tượng AudioSegment để lấy mẫu âm thanh
        raw_audio = AudioSegment.from_file(path)
        samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')

        # Kiểm tra độ dài của dữ liệu âm thanh
        if len(samples) < FRAME_LENGTH:
            print(f"Audio quá ngắn: {path}")
            return None, None

        # Cắt bỏ khoảng lặng ở đầu và cuối
        trimmed, _ = librosa.effects.trim(samples, top_db=25)

        # Đệm dữ liệu âm thanh để có độ dài cố định
        max_length = 180000  # Độ dài tối đa theo yêu cầu của bạn
        if len(trimmed) < max_length:
            padded = np.pad(trimmed, (0, max_length - len(trimmed)), 'constant')
        else:
            padded = trimmed[:max_length]  # Trường hợp âm thanh dài hơn max_length

        # Tải dữ liệu âm thanh và tần số lấy mẫu (nếu cần thiết)
        y, sr = librosa.load(path, sr=None)

        return padded, sr
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None, None

def decode(label_index):
    return reverse_emotion_dic.get(label_index)

def suggest_song(emotion):
    """
    Suggests a single song based on the predicted emotion, using rules based on audio features.
    Returns a dictionary containing the track_name, track_artist, image_url, and spotify_url of the selected song.
    Returns None if song_data is None or no suitable song is found.
    """
    if song_data is None:
        return None

    try:
        # Convert emotion string to lowercase for consistency
        emotion = emotion.lower()

        # Define rules based on emotion (adjust these rules as needed)
        if emotion == 'happy':
            # Suggest high energy and valence songs
            suitable_songs = song_data[(song_data['energy'] > 0.7) & (song_data['valence'] > 0.7)]
        elif emotion == 'sad':
            # Suggest low energy and valence songs
            suitable_songs = song_data[(song_data['energy'] < 0.4) & (song_data['valence'] < 0.4)]
        elif emotion == 'angry':
            # Suggest high energy and loudness songs
            suitable_songs = song_data[(song_data['energy'] > 0.8) & (song_data['loudness'] < -5)]
        elif emotion == 'fear':
             # Suggest low energy and acousticness songs
            suitable_songs = song_data[(song_data['energy'] < 0.5) & (song_data['acousticness'] > 0.5)]
        elif emotion == 'disgust':
            # Suggest high energy and low speechiness songs
            suitable_songs = song_data[(song_data['energy'] > 0.7) & (song_data['speechiness'] < 0.3)]

        else: # Neutral or unknown emotion
            # Suggest songs with moderate valence and energy
            suitable_songs = song_data[(song_data['valence'] > 0.4) & (song_data['valence'] < 0.6) & (song_data['energy'] > 0.4) & (song_data['energy'] < 0.6)]

        if not suitable_songs.empty:
            # Select a random song from the suitable songs
            selected_song = suitable_songs.sample(1)
            # Extract required information
            track_name = selected_song['track_name'].values[0] if 'track_name' in selected_song else "N/A"
            track_artist = selected_song['track_artist'].values[0] if 'track_artist' in selected_song else "N/A"
            image_url = selected_song['image_url'].values[0] if 'image_url' in selected_song else "N/A"
            spotify_url = selected_song['spotify_url'].values[0] if 'spotify_url' in selected_song else "N/A"


            return {
                'track_name': track_name,
                'track_artist': track_artist,
                'image_url': image_url,
                'spotify_url': spotify_url
            }
        else:
            print(f"No suitable songs found for emotion: {emotion}")
            return None # No suitable songs found

    except Exception as e:
        print(f"Error suggesting song: {e}")
        return None

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')  # Cho phép origin cụ thể
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization') #Cho phép header
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS') #Cho phép method
    return response


@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    file_path = './uploaded_audio.wav'
    audio_file.save(file_path)

    # Xử lý audio
    padded_audio, sr = preprocess_audio(file_path)
    if padded_audio is None:
        return jsonify({'error': 'Failed to process the audio file'}), 500

    # Tính toán các đặc trưng
    zcr = librosa.feature.zero_crossing_rate(padded_audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=padded_audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=padded_audio, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

    # Kết hợp các đặc trưng
    features = np.concatenate((
        np.swapaxes(zcr, 0, 1),  # Hoán đổi trục cho zcr và rms
        np.swapaxes(rms, 0, 1),
        np.swapaxes(mfccs, 0, 1)),
        axis=1
    )

    # Thêm chiều batch
    features = np.expand_dims(features, axis=0)
    features = features.astype('float32')

    # Dự đoán
    predictions = MODEL.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_emotion = decode(predicted_class[0])

    # Get song suggestions
    suggested_song = suggest_song(predicted_emotion)

    response_data = {
        'predicted_class': int(predicted_class[0]),
        'predicted_emotion': predicted_emotion,
    }

    if suggested_song:
        response_data['suggested_song'] = suggested_song
    else:
        response_data['suggested_song'] = None # Or a message "No song found"
    print(response_data)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)