from flask import Flask, request
from ttslearn.dnntts import DNNTTS
from huggingsound import SpeechRecognitionModel
import logging, os

app = Flask(__name__)
logger = logging.getLogger()

@app.route("/")
def index():
    return "HELLO WORLD"

@app.route("/stt", methods = ['POST'])
def stt():
    audio_file = request.files['file']
    file_name = "Audio.wav"
    audio_file.save(file_name)
    audio_paths = [file_name]
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-japanese")
    transcriptions = model.transcribe(audio_paths)
    
    os.remove(file_name)
    
    text = ' .'.join(list(t['transcription'] for t in transcriptions))
    
    return text

if __name__ == '__main__':
	app.run(debug=True)