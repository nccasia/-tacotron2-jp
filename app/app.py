from flask import Flask, request, send_file
from huggingsound import SpeechRecognitionModel
import logging, os
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
logger = logging.getLogger()

@app.route("/")
def index():
    return "HELLO WORLD"

#Speech-To-Text
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

#Text-To-Speech
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

#Sentiment Analyst
@app.route("/sa", methods = ['GET', 'POST'])
def sa():
    text = request.form.get('text', default='', type=str)
    tokenizer = AutoTokenizer.from_pretrained("jarvisx17/japanese-sentiment-analysis")

    model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Return the sentiment label with the highest probability
    label_map = {0: "negative", 1: "positive"}
    label_id = torch.argmax(logits, dim=1).tolist()[0]
    label = label_map[label_id]
    score = probabilities[label_id]

    return {"text": text, "sentiment": label, "score": score}

if __name__ == '__main__':
	app.run(debug=True)