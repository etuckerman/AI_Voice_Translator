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
import queue

# Initialize the recognizer
r = sr.Recognizer()

# Dictionary to map language names to their model codes
language_codes = {
    "English": "eng",
    "Hindi": "hin",
    "Bengali": "ben",
    "Tamil": "tam",
    "Telugu": "tel",
    "Marathi": "mar",
    "Gujarati": "guj",
    "Kannada": "kan",
    "Malayalam": "mal",
    "Punjabi": "pan",
    "Urdu": "urd-script_arabic"
}

# Initialize empty dictionary for lazy loading of TTS models
tts_models = {}

# Function to get or load TTS model
def get_tts_model(language):
    if language not in tts_models:
        model_name = f"facebook/mms-tts-{language_codes[language]}"
        tts_models[language] = (
            VitsModel.from_pretrained(model_name),
            AutoTokenizer.from_pretrained(model_name)
        )
    return tts_models[language]

def process_partial_speech(partial_text, selected_language, dest_language):
    try:
        translation_text = translate_text(partial_text, dest_language)
        if translation_text:
            output_text = f"Translation ({selected_language}): {translation_text}"
            output_label.config(text=output_text)
            transcript_text.insert(tk.END, f"Translation: {translation_text}\n")
            transcript_text.see(tk.END)

            status_label.config(text="Generating speech...")
            tts_model, tokenizer = get_tts_model(selected_language)

            inputs = tokenizer(translation_text, return_tensors="pt")
            with torch.no_grad():
                output = tts_model(**inputs).waveform

            status_label.config(text="Queueing translation...")
            speech_queue.put((output, selected_language))
    except Exception as e:
        print(f"Error in partial processing: {e}")
        
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

# Create a queue for speech outputs
speech_queue = queue.Queue()
is_speaking = False

# Function to handle the speech queue
def process_speech_queue():
    global is_speaking
    while True:
        try:
            if not is_speaking and not speech_queue.empty():
                is_speaking = True
                output, language = speech_queue.get()
                play_translation(output, language)
                is_speaking = False
            time.sleep(0.1)  # Small delay to prevent CPU overuse
        except Exception as e:
            print(f"Error in queue processing: {e}")
            is_speaking = False

# Function to play the translated audio
def play_translation(output, language):
    try:
        output = output.squeeze().numpy()
        output = output * 32767
        output = output.astype(np.int16)
        
        sd.play(output, samplerate=get_tts_model(language)[0].config.sampling_rate)
        sd.wait()
        status_label.config(text="Ready for more speech...")
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

# Add these global variables at the top with other initializations
is_listening = False
listening_thread = None

# Function to translate speech
def translate_speech():
    global is_listening
    
    while is_listening:
        with sr.Microphone() as source:
            status_label.config(text="Listening... Speak now.")
            r.adjust_for_ambient_noise(source)
            
            try:
                while is_listening:
                    audio = r.listen(source, phrase_time_limit=3)
                    
                    try:
                        partial_text = r.recognize_google(audio)
                        if partial_text:
                            update_transcript(partial_text)

                            selected_language = language_var.get()
                            dest_language = language_options[selected_language]

                            threading.Thread(
                                target=process_partial_speech,
                                args=(partial_text, selected_language, dest_language),
                                daemon=True
                            ).start()

                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        print(f"Error in recognition: {e}")
                        continue

            except Exception as e:
                if is_listening:  # Only show error if we're still supposed to be listening
                    status_label.config(text=f"Error: {str(e)}")
                    time.sleep(1)
    
    status_label.config(text="Stopped listening.")

# Function to start the translation in a separate thread
def start_translation():
    global is_listening, listening_thread
    
    if is_listening:
        # If already listening, stop the current session
        is_listening = False
        if listening_thread:
            listening_thread.join(timeout=1)
        start_button.config(text="Start Translation")
        status_label.config(text="Stopped listening.")
    else:
        # Start new listening session
        is_listening = True
        start_button.config(text="Stop Translation")
        output_label.config(text="")
        status_label.config(text="Starting translation...")
        listening_thread = threading.Thread(target=translate_speech, daemon=True)
        listening_thread.start()

# Function to exit the application
def exit_app():
    global is_listening
    is_listening = False
    if listening_thread:
        listening_thread.join(timeout=1)
    root.quit()

# Add this function after the other function definitions
def test_all_languages():
    test_button.config(state=tk.DISABLED)  # Disable button during test
    status_label.config(text="Starting language test...")
    
    def run_test():
        test_text = "This is a test"
        
        for language in language_codes.keys():
            try:
                status_label.config(text=f"Testing {language}...")
                
                # Translate "This is a test" to target language
                dest_code = language_options[language]
                translator = Translator()
                translation = translator.translate(test_text, dest=dest_code)
                
                # Update transcript
                transcript_text.insert(tk.END, f"\nTesting {language}: {translation.text}\n")
                transcript_text.see(tk.END)
                
                # Generate and queue speech
                tts_model, tokenizer = get_tts_model(language)
                inputs = tokenizer(translation.text, return_tensors="pt")
                with torch.no_grad():
                    output = tts_model(**inputs).waveform
                
                speech_queue.put((output, language))
                
            except Exception as e:
                transcript_text.insert(tk.END, f"Error testing {language}: {str(e)}\n")
                transcript_text.see(tk.END)
        
        status_label.config(text="Language test complete!")
        test_button.config(state=tk.NORMAL)  # Re-enable button
    
    # Run test in separate thread
    threading.Thread(target=run_test, daemon=True).start()
    
# Create and place the UI elements
instructions_label = tk.Label(root, text="Press 'Start Translation' and speak in any language.\nThe system will translate to your selected Indian language.", 
                            wraplength=500)
instructions_label.pack(pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Translation", command=start_translation)
start_button.pack(side=tk.LEFT, padx=5)

test_button = tk.Button(button_frame, text="Test All Languages", command=test_all_languages)
test_button.pack(side=tk.LEFT, padx=5)

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

# Preload the default language model (Hindi)
get_tts_model("Hindi")

# Start the queue processor
threading.Thread(target=process_speech_queue, daemon=True).start()



# Run the application
root.mainloop()