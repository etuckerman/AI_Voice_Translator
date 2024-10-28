#run myenv\Scripts\activate to run in env

import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import pyttsx3
from langdetect import detect
from googletrans import Translator
import threading

# Initialize the recognizer, translator, and TTS engine
r = sr.Recognizer()
translator = Translator(service_urls=['translate.google.com'])
tts = pyttsx3.init()

# Function to handle speech recognition and translation
def translate_speech():
    with sr.Microphone() as source:
        print("Speak now")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        input_language = detect(text)

        if input_language == "es":
            translation = translator.translate(text, dest='en')
            output_text = f"Spanish to English: {translation.text}"
        elif input_language == "en":
            translation = translator.translate(text, dest='es')
            output_text = f"English to Spanish: {translation.text}"
        else:
            output_text = "Unsupported language"

        # Speak the translated text
        tts.say(translation.text)
        tts.runAndWait()

        # Update the GUI with the output
        output_label.config(text=output_text)

    except sr.UnknownValueError:
        messagebox.showerror("Error", "Could not understand audio")
    except sr.RequestError as e:
        messagebox.showerror("Error", f"Could not request results; {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to start the translation in a separate thread
def start_translation():
    threading.Thread(target=translate_speech).start()

# Function to exit the application
def exit_app():
    root.quit()

# Create the main window
root = tk.Tk()
root.title("AI Voice Translator")

# Create and place the buttons and output label
start_button = tk.Button(root, text="Start Translation", command=start_translation)
start_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.pack(pady=20)

output_label = tk.Label(root, text="", wraplength=300)
output_label.pack(pady=20)

# Run the application
root.mainloop()
