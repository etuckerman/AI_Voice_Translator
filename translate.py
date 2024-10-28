import speech_recognition as sr
import pyttsx3
from langdetect import detect
from googletrans import Translator

#instance of speech input recognizer
r = sr.Recognizer()

#instance of text translator
translator = Translator(service_urls=['translate.google.com'])

#instance of text to speech, for translated to text -> speech
tts = pyttsx3.init()