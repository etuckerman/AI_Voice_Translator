import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from transformers import VitsModel, AutoTokenizer
import torch
import pyttsx3
from langdetect import detect
from googletrans import Translator
import threading
import scipy.io.wavfile
import numpy as np
import sounddevice as sd

# Initialize the recognizer, translator, and TTS engine
r = sr.Recognizer()
translator = Translator(service_urls=['translate.google.com'])

# Initialize the TTS model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

def play_translation(output):
    # Convert the output to a suitable format
    output = output.squeeze().numpy()  # Remove unnecessary dimensions
    output = output * 32767  # Scale to int16 range
    output = output.astype(np.int16)  # Convert to int16

    # Play the audio
    sd.play(output, samplerate=model.config.sampling_rate)
    sd.wait()  # Wait until the sound has finished playing

def update_transcript(text):
    # Update the transcript with the user's spoken text
    transcript_text.insert(tk.END, f"You: {text}\n")  # Append the user's input
    transcript_text.see(tk.END)  # Scroll to the end of the transcript

def translate_speech():
    while True:  # Loop to continuously listen for new input
        with sr.Microphone() as source:
            status_label.config(text="Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

            # Convert audio to text
            try:
                text = r.recognize_google(audio)
                update_transcript(text)  # Update the transcript with the user's input
                input_language = detect(text)

                # Initialize translation variables
                translation_text = ""
                detected_language = ""

                if input_language == "es":
                    translation = translator.translate(text, dest='en')
                    translation_text = translation.text
                    output_text = f"Spanish to English: {translation_text}"
                    detected_language = "Detected Language: Spanish"
                elif input_language == "en":
                    translation = translator.translate(text, dest='es')
                    translation_text = translation.text
                    output_text = f"English to Spanish: {translation_text}"
                    detected_language = "Detected Language: English"
                else:
                    output_text = "Unsupported language"
                    detected_language = "Detected Language: Unknown"

                # Check if translation is valid
                if translation_text:
                    # Update the GUI with the output
                    output_label.config(text=output_text)
                    language_label.config(text=detected_language)

                    # Update the transcript with the translation
                    transcript_text.insert(tk.END, f"Translation: {output_text}\n")  # Append the translation
                    transcript_text.see(tk.END)  # Scroll to the end of the transcript

                    # Speak the translated text using the new TTS model
                    inputs = tokenizer(translation_text, return_tensors="pt")
                    with torch.no_grad():
                        output = model(**inputs).waveform

                    # Start TTS playback in a separate thread
                    threading.Thread(target=play_translation, args=(output,), daemon=True).start()

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
                print(f"An error occurred: {e}")  # Print the error for debugging
                messagebox.showerror("Error", f"An error occurred: {e}")
                status_label.config(text="Error occurred")
                
                
# Function to start the translation in a separate thread
def start_translation():
    output_label.config(text="")
    language_label.config(text="")
    status_label.config(text="Starting translation...")
    threading.Thread(target=translate_speech, daemon=True).start()

# Function to exit the application
def exit_app():
    root.quit()

# Create the main window
root = tk.Tk()
root.title("AI Voice Translator")
root.geometry("400x400")  # Adjusted window size for transcript

# Create and place the buttons and output label
instructions_label = tk.Label(root, text="Press 'Start Translation' to begin speaking.", wraplength=300)
instructions_label.pack(pady=10)

start_button = tk.Button(root, text="Start Translation", command=start_translation)
start_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.pack(pady=10)

output_label = tk.Label(root, text="", wraplength=300)
output_label.pack(pady=10)

language_label = tk.Label(root, text="", wraplength=300)  # Label to show detected language
language_label.pack(pady=10)

status_label = tk.Label(root, text="", wraplength=300)
status_label.pack(pady=10)

# Add a Text widget for the transcript
transcript_text = tk.Text(root, wrap=tk.WORD, height=10, width=50)
transcript_text.pack(pady=10)

# Run the application
root.mainloop()