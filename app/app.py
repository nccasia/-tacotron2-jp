from flask import Flask, request, send_file
from huggingsound import SpeechRecognitionModel
import logging, os
from gtts import gTTS

app = Flask(__name__)
logger = logging.getLogger()

@app.route("/")
def index():
    return "HELLO WORLD"

@app.route("/stt", methods = ['POST', 'GET'])
def stt():
    models = {
        "japanese" : "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "vietnamese" : "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        "english" : "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    }
    language = request.args.get('language', default='japanese', type=str)
    audio_file = request.files['file']
    file_name = "Audio.wav"
    audio_file.save(file_name)
    audio_paths = [file_name]
    try:
        model = SpeechRecognitionModel(models[language])
        transcriptions = model.transcribe(audio_paths)
        text = ' .'.join(list(t['transcription'] for t in transcriptions))
    except Exception:
        return "Sorry!! We don't support this language yet T_T"
    finally:
        os.remove(file_name)
    
    return text

@app.route("/tts", methods = ['GET', 'POST'])
def tts():
    langs = {
        "japanese" : "ja",
        "vietnamese" : "vi",
        "english" : "en"
    }
    language = request.args.get('language', default='japanese', type=str)
    text = request.form.get('text', default='', type=str)
    
    filename = "Audio.wav"
    
    if(os.path.exists(filename)):
        os.remove(filename)

    try:
        output = gTTS(text,lang=langs[language], slow=False)
        output.save(filename)
    except Exception:
        return "Sorry!! We don't support this language yet T_T"
        
    return send_file(filename)

if __name__ == '__main__':
	app.run(debug=True)