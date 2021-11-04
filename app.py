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
    indication = ""

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

        # # THIS FOR TEST AUDIO FILES
        # audioFile = sr.AudioFile('bonafide3.flac')
        # with audioFile as source:
        #     recognizer.adjust_for_ambient_noise(source)
        #     data = recognizer.record(source)

        # THIS FOR MICROPHONE INPUT
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            indicaction = 'Start recording'
            data = recognizer.listen(source, 6, 4)
            indication = 'End recording'
            with open('speech.flac', 'wb') as f:
                f.write(data.get_flac_data())

        # detect = spoofdetect_stack('speech.flac','test/LA_aug.pth','test/PA_noise.pth')

        detect = spoofdetect('speech.flac','test/SA_abs_10000.pth')

        if detect[0]>detect[1]:
            is_spoof = 'Spoof!'
            confidence = detect[0]*100
        else:
            is_spoof = 'Bonafide'
            confidence = detect[1]*100

        if is_spoof == 'Bonafide':
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

    return render_template('index.html', transcript=transcript, is_spoof=is_spoof, confidence=confidence, indication=indication)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
