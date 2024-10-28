import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from transformers import VitsModel, AutoTokenizer
import torch
import requests
import threading
import numpy as np
import sounddevice as sd
import time

# Initialize the recognizer
r = sr.Recognizer()

# Initialize TTS models and tokenizers for different languages
tts_models = {
    "English": (VitsModel.from_pretrained("facebook/mms-tts-eng"), AutoTokenizer.from_pretrained("facebook/mms-tts-eng")),
    "Tagalog": (VitsModel.from_pretrained("facebook/mms-tts-tgl"), AutoTokenizer.from_pretrained("facebook/mms-tts-tgl")),
    "Bengali": (VitsModel.from_pretrained("facebook/mms-tts-ben"), AutoTokenizer.from_pretrained("facebook/mms-tts-ben")),
    "Spanish": (VitsModel.from_pretrained("facebook/mms-tts-spa"), AutoTokenizer.from_pretrained("facebook/mms-tts-spa")),
}

# Function to play the translated audio
def play_translation(output, language):
    output = output.squeeze().numpy()
    output = output * 32767
    output = output.astype(np.int16)
    sd.play(output, samplerate=tts_models[language][0].config.sampling_rate)
    sd.wait()

def update_transcript(text):
    transcript_text.insert(tk.END, f"You: {text}\n")
    transcript_text.see(tk.END)

# Function to translate text using DeepL API
def deepl_translate(text, dest_language):
    api_key = "af83cb0b-b297-4dfb-98f1-5f1b9bd991c7:fx"  # Replace with your DeepL API key
    url = "https://api-free.deepl.com/v2/translate"
    params = {
        'auth_key': api_key,
        'text': text,
        'target_lang': dest_language.upper()
    }
    
    response = requests.post(url, data=params)
    if response.status_code == 200:
        return response.json()['translations'][0]['text']
    else:
        print(f"Translation failed: {response.status_code} {response.text}")  # Print the response text for debugging
        return None

# Create the main window
root = tk.Tk()
root.title("AI Voice Translator")
root.geometry("400x400")

# Create a dropdown menu for language selection
language_var = tk.StringVar()
language_var.set("English")  # Default value

# List of languages for the dropdown with correct language codes
language_options = {
    "English": "EN",
    "Tagalog": "TL",  # Check if DeepL supports this language
    "Bengali": "BN",
    "Spanish": "ES"
}

# Create the dropdown menu
language_menu = tk.OptionMenu(root, language_var, *language_options.keys())
language_menu.pack(pady=10)

# Function to translate speech
def translate_speech():
    while True:
        with sr.Microphone() as source:
            status_label.config(text="Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                update_transcript(text)

                # Get the selected target language code from the dropdown
                selected_language = language_var.get()
                dest_language = language_options[selected_language]  # Get the corresponding code

                # Attempt to translate with retries
                translation_attempts = 0
                max_attempts = 5
                translation_success = False

                while translation_attempts < max_attempts and not translation_success:
                    try:
                        status_label.config(text=f"Reconnecting to DeepL...")
                        translation_text = deepl_translate(text, dest_language)

                        if translation_text:
                            translation_success = True
                        else:
                            raise ValueError("Translation returned None or invalid response.")

                    except Exception as e:
                        translation_attempts += 1
                        print(f"Attempt {translation_attempts} failed: {e}")
                        time.sleep(1)  # Wait for 1 second before retrying
                        if translation_attempts < max_attempts:
                            continue
                        else:
                            status_label.config(text="Translation failed after multiple attempts.")
                            break

                if translation_success and translation_text:
                    output_text = f"Translation to {selected_language}: {translation_text}"
                    output_label.config(text=output_text)

                    transcript_text.insert(tk.END, f"Translation: {output_text}\n")
                    transcript_text.see(tk.END)

                    # Get the appropriate TTS model and tokenizer for the selected language
                    tts_model, tokenizer = tts_models[selected_language]

                    inputs = tokenizer(translation_text, return_tensors="pt")
                    with torch.no_grad():
                        output = tts_model(**inputs).waveform

                    threading.Thread(target=play_translation, args=(output, selected_language), daemon=True).start()
                    status_label.config(text="Translation complete.")
                else:
                    print("Translation is empty or failed.")
                    status_label.config(text="Translation failed.")

            except sr.UnknownValueError:
                messagebox.showerror("Error", "Could not understand audio")
                status_label.config(text="Error: Could not understand audio")
            except sr.RequestError as e:
                messagebox.showerror("Error", f"Could not request results; {e}")
                status_label.config(text="Error: Request failed")
            except Exception as e:
                print(f"An error occurred: {e}")
                messagebox.showerror("Error", f"An error occurred: {e}")
                status_label.config(text="Error occurred")

# Function to start the translation in a separate thread
def start_translation():
    output_label.config(text="")
    status_label.config(text="Starting translation...")
    threading.Thread(target=translate_speech, daemon=True).start()

# Function to exit the application
def exit_app():
    root.quit()

# Create and place the buttons and output label
instructions_label = tk.Label(root, text="Press 'Start Translation' to begin speaking.", wraplength=300)
instructions_label.pack(pady=10)

start_button = tk.Button(root, text="Start Translation", command=start_translation)
start_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.pack(pady=10)

output_label = tk.Label(root, text="", wraplength=300)
output_label.pack(pady=10)

status_label = tk.Label(root, text="", wraplength=300)
status_label.pack(pady=10)

# Add a Text widget for the transcript
transcript_text = tk.Text(root, wrap=tk.WORD, height=10, width=50)
transcript_text.pack(pady=10)

# Run the application
root.mainloop()