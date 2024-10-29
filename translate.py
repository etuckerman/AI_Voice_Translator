import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from transformers import VitsModel, AutoTokenizer
import torch
import threading
import numpy as np
import sounddevice as sd
import time
from googletrans import Translator

# Initialize the recognizer
r = sr.Recognizer()

# Initialize TTS models and tokenizers for different languages
tts_models = {
    "English": (VitsModel.from_pretrained("facebook/mms-tts-eng"), AutoTokenizer.from_pretrained("facebook/mms-tts-eng")),
    "Hindi": (VitsModel.from_pretrained("facebook/mms-tts-hin"), AutoTokenizer.from_pretrained("facebook/mms-tts-hin")),
    "Bengali": (VitsModel.from_pretrained("facebook/mms-tts-ben"), AutoTokenizer.from_pretrained("facebook/mms-tts-ben")),
    "Tamil": (VitsModel.from_pretrained("facebook/mms-tts-tam"), AutoTokenizer.from_pretrained("facebook/mms-tts-tam")),
    "Telugu": (VitsModel.from_pretrained("facebook/mms-tts-tel"), AutoTokenizer.from_pretrained("facebook/mms-tts-tel")),
    "Marathi": (VitsModel.from_pretrained("facebook/mms-tts-mar"), AutoTokenizer.from_pretrained("facebook/mms-tts-mar")),
    "Gujarati": (VitsModel.from_pretrained("facebook/mms-tts-guj"), AutoTokenizer.from_pretrained("facebook/mms-tts-guj")),
    "Kannada": (VitsModel.from_pretrained("facebook/mms-tts-kan"), AutoTokenizer.from_pretrained("facebook/mms-tts-kan")),
    "Malayalam": (VitsModel.from_pretrained("facebook/mms-tts-mal"), AutoTokenizer.from_pretrained("facebook/mms-tts-mal")),
    "Punjabi": (VitsModel.from_pretrained("facebook/mms-tts-pan"), AutoTokenizer.from_pretrained("facebook/mms-tts-pan")),
    "Urdu": (VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic"), AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic"))
}

# List of languages for the dropdown with correct language codes
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# Create the main window
root = tk.Tk()
root.title("Indian Language Voice Translator")
root.geometry("600x800")  # Increased size to accommodate more languages

# Function to play the translated audio
def play_translation(output, language):
    try:
        output = output.squeeze().numpy()
        output = output * 32767
        output = output.astype(np.int16)
        sd.play(output, samplerate=tts_models[language][0].config.sampling_rate)
        sd.wait()
        status_label.config(text="Playback complete.")
    except Exception as e:
        status_label.config(text=f"Audio playback error: {str(e)}")

def update_transcript(text):
    transcript_text.insert(tk.END, f"You: {text}\n")
    transcript_text.see(tk.END)

# Function to translate text using Google Translate
def translate_text(text, dest_language):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

# Create a frame for the language selection
language_frame = tk.Frame(root)
language_frame.pack(pady=10)

# Create a label for the language dropdown
language_label = tk.Label(language_frame, text="Select Target Language:")
language_label.pack(side=tk.LEFT, padx=5)

# Create the language dropdown
language_var = tk.StringVar()
language_var.set("Hindi")  # Default value
language_menu = tk.OptionMenu(language_frame, language_var, *language_options.keys())
language_menu.pack(side=tk.LEFT)

# Function to translate speech
def translate_speech():
    while True:
        with sr.Microphone() as source:
            status_label.config(text="Listening... Speak now.")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                update_transcript(text)

                selected_language = language_var.get()
                dest_language = language_options[selected_language]

                status_label.config(text="Translating...")
                translation_text = translate_text(text, dest_language)

                if translation_text:
                    output_text = f"Translation ({selected_language}): {translation_text}"
                    output_label.config(text=output_text)
                    transcript_text.insert(tk.END, f"Translation: {translation_text}\n")
                    transcript_text.see(tk.END)

                    status_label.config(text="Generating speech...")
                    tts_model, tokenizer = tts_models[selected_language]

                    inputs = tokenizer(translation_text, return_tensors="pt")
                    with torch.no_grad():
                        output = tts_model(**inputs).waveform

                    status_label.config(text="Playing translation...")
                    threading.Thread(target=play_translation, args=(output, selected_language), daemon=True).start()
                else:
                    status_label.config(text="Translation failed. Please try again.")

            except sr.UnknownValueError:
                status_label.config(text="Could not understand audio. Please try again.")
            except sr.RequestError as e:
                status_label.config(text=f"Speech recognition error: {str(e)}")
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}")

# Function to start the translation in a separate thread
def start_translation():
    output_label.config(text="")
    status_label.config(text="Starting translation...")
    threading.Thread(target=translate_speech, daemon=True).start()

# Function to exit the application
def exit_app():
    root.quit()

# Create and place the UI elements
instructions_label = tk.Label(root, text="Press 'Start Translation' and speak in any language.\nThe system will translate to your selected Indian language.", 
                            wraplength=500)
instructions_label.pack(pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Translation", command=start_translation)
start_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Exit", command=exit_app)
exit_button.pack(side=tk.LEFT, padx=5)

# Create labels for output and status
output_label = tk.Label(root, text="", wraplength=500)
output_label.pack(pady=10)

status_label = tk.Label(root, text="", wraplength=500)
status_label.pack(pady=10)

# Create a frame for the transcript
transcript_frame = tk.Frame(root)
transcript_frame.pack(pady=10, fill=tk.BOTH, expand=True)

transcript_label = tk.Label(transcript_frame, text="Conversation History:")
transcript_label.pack()

transcript_text = tk.Text(transcript_frame, wrap=tk.WORD, height=15, width=60)
transcript_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Add a scrollbar to the transcript
scrollbar = tk.Scrollbar(transcript_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

transcript_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=transcript_text.yview)

# Run the application
root.mainloop()