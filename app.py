from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import playsound
from model_main_test import *

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    is_spoof = ""
    confidence = 0

    def speak(text):
        tts = gTTS(text=text, lang='zh-cn')

        filename = "translated.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        # os.remove(filename)
        return

    if request.method == "POST":
        print("FORM DATA RECEIVED")

        recognizer = sr.Recognizer()
        trans = Translator()
        # audioFile = sr.AudioFile('speech.wav')
        # with audioFile as source:
        #     recognizer.adjust_for_ambient_noise(source)
        #     data = recognizer.record(source)
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            data = recognizer.listen(source, timeout=4)
            with open('speech.wav', 'wb') as f:
                f.write(data.get_wav_data())

        detect = spoofdetect('speech.wav','test/SA.pth')

        if detect[0]>detect[1]:
            is_spoof = 'Spoof!'
            confidence = detect[0]*100
        else:
            is_spoof = 'Bonafide'
            confidence = detect[1]*100

        try:
            srctext = recognizer.recognize_google(data)
            transcript = trans.translate(srctext, dest='zh-cn', src='en').text
            speak(transcript)

            with open('srctext.txt', 'w') as file:
                file.write(srctext)

            with open('desttext.txt', 'w') as file:
                file.write(transcript)

        except:
            transcript = 'I dont understand.... or something goes wrong'

    return render_template('index.html', transcript=transcript, is_spoof=is_spoof, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
