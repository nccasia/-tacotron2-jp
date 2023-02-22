from flask import Flask, request, send_file
from ttslearn.dnntts import DNNTTS
from huggingsound import SpeechRecognitionModel
import logging, os
import soundfile as sf
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

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
    model = SpeechRecognitionModel(models[language])
    transcriptions = model.transcribe(audio_paths)
    
    os.remove(file_name)
    
    text = ' .'.join(list(t['transcription'] for t in transcriptions))
    
    return text

@app.route("/tts", methods = ['GET', 'POST'])
def tts():
    language = request.args.get('language', default='japanese', type=str)
    text = request.form.get('text', default='', type=str)
    
    filename = "Audio.wav"
    
    try:
        os.remove(filename)
    except Exception:
        pass
    
    if language == 'japanese':
        print("Text-to-Speech JP")
        engine = DNNTTS()
        wav, sr = engine.tts(text)
        sf.write(filename, wav, sr)
        
    elif language == 'english':
        print("Text-to-Speech EN")
        
        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")

        mel_output, mel_length, alignment = tacotron2.encode_text(text)

        waveforms = hifi_gan.decode_batch(mel_output)
        
        torchaudio.save(filename, waveforms.squeeze(1), 22050)
        
    else:
        return "Sorry!! We don't support this language yet T_T"
        
    return send_file(filename)

if __name__ == '__main__':
	app.run(debug=True)