import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from transformers import VitsModel, AutoTokenizer
import torch
import threading
from googletrans import Translator

# Initialize the recognizer, translator, and TTS model
r = sr.Recognizer()
translator = Translator(service_urls=['translate.google.com'])
model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

# Create the main window
root = tk.Tk()
root.title("AI Voice Translator")
root.geometry("400x400")

# Create and place the buttons and output label
instructions_label = tk.Label(root, text="Press 'Start Translation' to begin.", wraplength=300)
instructions_label.pack(pady=10)

start_button = tk.Button(root, text="Start Translation", command=lambda: threading.Thread(target=start_translation, daemon=True).start())
start_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=10)

output_label = tk.Label(root, text="", wraplength=300)
output_label.pack(pady=10)

language_label = tk.Label(root, text="", wraplength=300)
language_label.pack(pady=10)

status_label = tk.Label(root, text="", wraplength=300)
status_label.pack(pady=10)

# Add a Text widget for the transcript
transcript_text = tk.Text(root, wrap=tk.WORD, height=10, width=50)
transcript_text.pack(pady=10)

def play_translation(output):
    output = output.squeeze().numpy()  # Remove unnecessary dimensions
    output = output * 32767  # Scale to int16 range
    output = output.astype(np.int16)  # Convert to int16
    sd.play(output, samplerate=model.config.sampling_rate)
    sd.wait()  # Wait until the sound has finished playing

def update_transcript(text):
    transcript_text.insert(tk.END, f"Caller: {text}\n")  # Append the caller's input
    transcript_text.see(tk.END)  # Scroll to the end of the transcript
    
def capture_desktop_audio():
    fs = 44100  # Sample rate
    duration = 10  # Duration to listen (in seconds)

    # Record audio from the system output
    print("Listening for Spanish speaker...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64', blocking=True)
    sd.wait()  # Wait until the recording is finished

    # Convert the audio data to a format that can be recognized
    audio_data_int16 = np.int16(audio_data * 32767)  # Scale to int16 range
    audio = sr.AudioData(audio_data_int16.tobytes(), fs, 1)  # Create AudioData object

    # Play back the recorded audio for verification
    print("Playing back the recorded audio...")
    sd.play(audio_data_int16, samplerate=fs)
    sd.wait()  # Wait until the playback is finished

    try:
        print("Recognizing...")
        text = r.recognize_google(audio, language='es-ES')
        print(f"Recognized text: {text}")  # Debugging output
        if text:  # Check if recognized text is not empty
            update_transcript(f"Caller: {text}")  # Update the transcript with the caller's input

            # Translate Spanish to English
            translation = translator.translate(text, dest='en')
            output_text = f"Translation: {translation.text}"
            output_label.config(text=output_text)  # Update the GUI with the output

            # Speak the translated text
            inputs = tokenizer(translation.text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            threading.Thread(target=play_translation, args=(output,), daemon=True).start()

            status_label.config(text="Translation complete.")
        else:
            print("Recognized text is empty.")  # Debugging output
            messagebox.showerror("Error", "No speech recognized.")
            status_label.config(text="Error: No speech recognized.")

    except sr.UnknownValueError:
        print("Could not understand audio")  # Debugging output
        messagebox.showerror("Error", "Could not understand audio")
        status_label.config(text="Error: Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")  # Debugging output
        messagebox.showerror("Error", f"Could not request results; {e}")
        status_label.config(text="Error: Request failed")
    except Exception as e:
        print(f"An error occurred: {e}")  # Debugging output
        messagebox.showerror("Error", f"An error occurred: {e}")
        status_label.config(text="Error occurred")

def start_translation():
    # Start capturing desktop audio
    capture_desktop_audio()

def speak_response():
    # Capture your response in English
    with sr.Microphone() as source:
        status_label.config(text="Listening for your response...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

        try:
            # Recognize your English speech
            text = r.recognize_google(audio, language='en-US')
            update_transcript(f"You: {text}")  # Update the transcript with your input

            # Translate English to Spanish
            translation = translator.translate(text, dest='es')
            output_text = f"Translation: {translation.text}"
            output_label.config(text=output_text)  # Update the GUI with the output

            # Speak the translated text
            inputs = tokenizer(translation.text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            threading.Thread(target=play_translation, args=(output,), daemon=True).start()

            status_label.config(text="Translation complete.")

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

# Add a button to speak your response
response_button = tk.Button(root, text="Speak Response", command=lambda: threading.Thread(target=speak_response, daemon=True).start())
response_button.pack(pady=10)

# Run the application
root.mainloop()