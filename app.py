from flask import Flask, render_template, request
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os

from utils.model_main import *

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    is_spoof = ""
    confidence = 0
    filepath = 'audio_and_text/'
    def speak(text):
        tts = gTTS(text=text, lang='zh-cn')

        filename ='audio_and_text/translated.mp3'
        tts.save(filename)
        try:
            playsound.playsound(filename)
        except:
            print('please read README.md to make modification if encounter playsound error')
        # os.remove(filename)
        return

    if request.method == "POST":
        print("FORM DATA RECEIVED")

        recognizer = sr.Recognizer()
        trans = Translator()

        '''UNCOMMENT TO RECOGNIZE A INPUT AUDIO FILE'''
        # audioFile = sr.AudioFile('LA_spoof_sample.flac')
        # with audioFile as source:
        #     recognizer.adjust_for_ambient_noise(source)
        #     data = recognizer.record(source)
        #     with open('speech.wav', 'wb') as f:
        #         f.write(data.get_wav_data())

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            data = recognizer.listen(source, phrase_time_limit=4)
            with open(filepath+'speech.wav', 'wb') as f:
                f.write(data.get_wav_data())

        #detect = spoofdetect_voting('speech.wav','utils/models/PA_abs_aug.pth','utils/models/LA_abs_aug.pth', 'test/SA_abs_aug.pth')
        detect = spoofdetect(filepath+'speech.wav','utils/models/SA_augmentation.pth')
        print(detect)

        if detect[0]>detect[1]:
            is_spoof = 'Spoof!'
            confidence = detect[0]*100
        else:
            is_spoof = 'Bonafide'
            confidence = detect[1]*100

        if is_spoof == 'Bonafide':
            try:
                srctext = recognizer.recognize_google(data, language='en-en')
                print(srctext)
                transcript = trans.translate(srctext, dest='zh-cn', src='en').text
                print(transcript)
                speak(transcript)

                with open(filepath+'srctext.txt', 'w') as file:
                    file.write(srctext)

                with open(filepath+'desttext.txt', 'w') as file:
                    file.write(transcript)

            except Exception:
                import  traceback
                traceback.print_exc()

                transcript = 'I dont understand....'

    return render_template('index.html', transcript=transcript, is_spoof=is_spoof, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
