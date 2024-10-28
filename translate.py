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


#loop permanently
while True:
    
    #use default mic 
    with sr.Microphone() as source:
        print("Speak now")
        
        #calibrate mic for ambient noise through mic
        r.adjust_for_ambient_noise(source)
        
        #listen for audio input from user
        audio = r.listen(source)
        
    try:
        #recognize speech using google speech recog
        text = r.recognize_google(audio)
        
        #detect lang
        input_language = detect(text)
        
        #LANGUAGE LIST:
        
        #spanish
        if input_language == "es":
            translation = translator.translate(text, dest='en')
            print("Spanish to English: {translation.text}")

            #speak the text
            tts.say(translation.text)
            tts.runAndWait()
        
        #translate speech to spanish if detected lang is english
        elif input_language == "en":
            translation = translator.translate(text, dest='es')
            print(f"English to Spanish: {translation.text}")
            
            #speak the text
            tts.say(translation.text)
            tts.runAndWait()
        else:
            print("Unsupported language")
        
    #GSR cant understand the audio
    except sr.UnknownValueError:
        print("GSR could not understand audio")
        
    #incase of problem with GSR servers
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        
        
                    
            
        
    
