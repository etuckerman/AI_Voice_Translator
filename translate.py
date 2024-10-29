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
    # Hispanic/Latino Languages (Most common in USA)
    "Spanish": "spa",
    "Portuguese": "por",
    
    # Asian Languages
    "Vietnamese": "vie",
    "Korean": "kor",
    "Tagalog": "tgl",
    
    # Middle Eastern Languages
    "Arabic": "ara",
    "Farsi": "fas",
    
    # South Asian Languages
    "Hindi": "hin",
    "Bengali": "ben",
    "Urdu": "urd-script_arabic",
    "Punjabi": "pan",
    "Tamil": "tam",
    "Telugu": "tel",
    "Gujarati": "guj",
    
    # European Languages
    "Russian": "rus",
    "Ukrainian": "ukr",
    "Polish": "pol",
    "English": "eng",
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

            # Special handling for Korean
            if selected_language == "Korean":
                inputs = tokenizer(translation_text, return_tensors="pt")
                # Convert float tensors to long tensors for Korean
                inputs = {k: v.long() if torch.is_floating_point(v) else v 
                         for k, v in inputs.items()}
            else:
                inputs = tokenizer(translation_text, return_tensors="pt")

            with torch.no_grad():
                output = tts_model(**inputs).waveform

            status_label.config(text="Queueing translation...")
            speech_queue.put((output, selected_language))
    except Exception as e:
        print(f"Error in partial processing: {e}")
        
# List of languages for the dropdown with correct language codes
language_options = {
    # Asian Languages
    "Chinese (Mandarin)": "zh-cn",
    "Vietnamese": "vi",
    "Korean": "ko",
    "Tagalog": "tl",
    "Japanese": "ja",
    
    # Hispanic/Latino Languages
    "Spanish": "es",
    "Portuguese": "pt",
    
    # South Asian Languages
    "Hindi": "hi",
    "Bengali": "bn",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    
    # Middle Eastern Languages
    "Arabic": "ar",
    "Farsi": "fa",
    
    # European Languages
    "Russian": "ru",
    "Ukrainian": "uk",
    "Polish": "pl",
    "English": "en",
}

# Modify the create_language_menu function to use the update_language_selection function
def create_language_menu(parent, variable):
    menu = tk.Menu(parent, tearoff=0)
    
    # Hispanic/Latino Languages (prioritized)
    hispanic_menu = tk.Menu(menu, tearoff=0)
    for lang in ["Spanish", "Portuguese"]:
        hispanic_menu.add_radiobutton(label=lang, variable=variable, value=lang, command=lambda lang=lang: update_language_selection(lang))
    menu.add_cascade(label="Hispanic/Latino Languages", menu=hispanic_menu)
    
    # Asian Languages
    asian_menu = tk.Menu(menu, tearoff=0)
    for lang in ["Vietnamese", "Korean", "Tagalog"]:
        asian_menu.add_radiobutton(label=lang, variable=variable, value=lang, command=lambda lang=lang: update_language_selection(lang))
    menu.add_cascade(label="Asian Languages", menu=asian_menu)
    
    # South Asian Languages
    south_asian_menu = tk.Menu(menu, tearoff=0)
    for lang in ["Hindi", "Bengali", "Urdu", "Punjabi", "Tamil", "Telugu", "Gujarati"]:
        south_asian_menu.add_radiobutton(label=lang, variable=variable, value=lang, command=lambda lang=lang: update_language_selection(lang))
    menu.add_cascade(label="South Asian Languages", menu=south_asian_menu)
    
    # Middle Eastern Languages
    middle_eastern_menu = tk.Menu(menu, tearoff=0)
    for lang in ["Arabic", "Farsi"]:
        middle_eastern_menu.add_radiobutton(label=lang, variable=variable, value=lang, command=lambda lang=lang: update_language_selection(lang))
    menu.add_cascade(label="Middle Eastern Languages", menu=middle_eastern_menu)
    
    # European Languages
    european_menu = tk.Menu(menu, tearoff=0)
    for lang in ["Russian", "Ukrainian", "Polish", "English"]:
        european_menu.add_radiobutton(label=lang, variable=variable, value=lang, command=lambda lang=lang: update_language_selection(lang))
    menu.add_cascade(label="European Languages", menu=european_menu)
    
    return menu

# Create the main window
root = tk.Tk()
# Change the title of the application
root.title("Multilingual Voice Translator")
root.geometry("600x800")  # Increased size to accommodate more languages

# Add a loading label to the GUI
loading_label = tk.Label(root, text="Loading models, please wait...", wraplength=500, fg="blue")

loading_label.pack(pady=10)

# Function to preload all TTS models
def preload_models():
    for language in language_codes.keys():
        try:
            get_tts_model(language)
            loading_label.config(text=f"Loaded model for {language}...")
        except Exception as e:
            print(f"Error loading model for {language}: {e}")
            loading_label.config(text=f"Error loading model for {language}")
    
    # Update the loading label to indicate completion
    loading_label.config(text="All models loaded successfully!")


# Start preloading models in a separate thread
threading.Thread(target=preload_models, daemon=True).start()

# Update the process_partial_speech function to remove delays
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

            # Special handling for Korean
            if selected_language == "Korean":
                inputs = tokenizer(translation_text, return_tensors="pt")
                inputs = {k: v.long() if torch.is_floating_point(v) else v 
                          for k, v in inputs.items()}
            else:
                inputs = tokenizer(translation_text, return_tensors="pt")

            with torch.no_grad():
                output = tts_model(**inputs).waveform

            status_label.config(text="Queueing translation...")
            speech_queue.put((output, selected_language))
    except Exception as e:
        print(f"Error in partial processing: {e}")


# Add this label after creating the root window
current_audio_label = tk.Label(root, text="Currently playing: None", wraplength=500)
current_audio_label.pack(pady=5)

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
                current_audio_label.config(text=f"Currently playing: {language}")
                play_translation(output, language)
                current_audio_label.config(text="Currently playing: None")
                is_speaking = False
            time.sleep(0.1)  # Small delay to prevent CPU overuse
        except Exception as e:
            print(f"Error in queue processing: {e}")
            is_speaking = False
            current_audio_label.config(text="Currently playing: None")

# Function to play the translated audio
def play_translation(output, language):
    try:
        output = output.squeeze().numpy()
        
        # Apply voice modifications
        # Speed modification (resample the audio)
        if speed_var.get() != 1.0:
            original_length = len(output)
            new_length = int(original_length / speed_var.get())
            indices = np.linspace(0, original_length-1, new_length)
            output = np.interp(indices, np.arange(original_length), output)
        
        # Pitch modification (basic implementation)
        if pitch_var.get() != 1.0:
            # We'll modify the sample rate instead of actual pitch
            sample_rate = int(get_tts_model(language)[0].config.sampling_rate * pitch_var.get())
        else:
            sample_rate = get_tts_model(language)[0].config.sampling_rate
        
        # Volume modification
        output = output * volume_var.get()
        
        # Convert to 16-bit integer format
        output = output * 32767
        output = output.astype(np.int16)
        
        sd.play(output, samplerate=sample_rate)
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
    
# Update the language selection button text when a language is selected
def update_language_selection(value):
    language_var.set(value)
    status_label.config(text=f"Language set to: {value}")

    
# Create a frame for the language selection
language_frame = tk.Frame(root)
language_frame.pack(pady=10)

# Create a label for the language dropdown
language_label = tk.Label(language_frame, text="Select Target Language:")
language_label.pack(side=tk.LEFT, padx=5)

# Create the language dropdown with groups
language_var = tk.StringVar()
language_var.set("Select Language")  # Default value
language_button = tk.Menubutton(language_frame, textvariable=language_var, relief=tk.RAISED)
language_button.pack(side=tk.LEFT)
language_button.menu = create_language_menu(language_button, language_var)
language_button["menu"] = language_button.menu


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
    if messagebox.askokcancel("Quit", "Do you really want to quit?"):
        root.quit()


def test_current_language():
    test_button_current.config(state=tk.DISABLED)  # Disable button during test
    status_label.config(text="Testing current language...")
    
    def run_single_test():
        test_text = "The quick brown fox jumps over the lazy dog, while five boxing wizards jump quickly."
        selected_language = language_var.get()
        
        try:
            # Translate test text to target language
            dest_code = language_options[selected_language]
            translator = Translator()
            translation = translator.translate(test_text, dest=dest_code)
            
            # Update transcript
            transcript_text.insert(tk.END, f"\nTesting {selected_language}: {translation.text}\n")
            transcript_text.see(tk.END)
            
            # Generate and queue speech
            tts_model, tokenizer = get_tts_model(selected_language)
            inputs = tokenizer(translation.text, return_tensors="pt")
            with torch.no_grad():
                output = tts_model(**inputs).waveform
            
            speech_queue.put((output, selected_language))
            
        except Exception as e:
            transcript_text.insert(tk.END, f"Error testing {selected_language}: {str(e)}\n")
            transcript_text.see(tk.END)
        
        status_label.config(text="Current language test complete!")
        test_button_current.config(state=tk.NORMAL)  # Re-enable button
    
    # Run test in separate thread
    threading.Thread(target=run_single_test, daemon=True).start()

# Add this function after the other function definitions
def test_all_languages():
    test_button.config(state=tk.DISABLED)  # Disable button during test
    status_label.config(text="Starting language test...")
    
    def run_test():
        test_text = "The quick brown fox jumps over the lazy dog, while five boxing wizards jump quickly."
        
        for language in language_codes.keys():
            try:
                status_label.config(text=f"Testing {language}...")
                
                # Translate "The quick brown fox jumps over the lazy dog, while five boxing wizards jump quickly." to target language
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
instructions_label = tk.Label(root, text="Press 'Start Translation'...", font=("Helvetica", 12, "bold"), wraplength=500)

instructions_label.pack(pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Translation", command=start_translation)
start_button.pack(side=tk.LEFT, padx=5)

test_button = tk.Button(button_frame, text="Test All Languages", command=test_all_languages)
test_button.pack(side=tk.LEFT, padx=5)

test_button_current = tk.Button(button_frame, text="Test Current Language", command=test_current_language)
test_button_current.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Exit", command=exit_app)
exit_button.pack(side=tk.LEFT, padx=5)

# Create labels for output and status
output_label = tk.Label(root, text="", wraplength=500)
output_label.pack(pady=10)

status_label = tk.Label(root, text="", wraplength=500, fg="green")

status_label.pack(pady=10)

# Create a frame for the transcript
transcript_frame = tk.Frame(root)
transcript_frame.pack(pady=10, fill=tk.BOTH, expand=True)

transcript_label = tk.Label(transcript_frame, text="Conversation History:")
transcript_label.pack()

transcript_text = tk.Text(transcript_frame, wrap=tk.WORD, height=15, width=60)
transcript_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Add a scrollbar to the transcript
scrollbar = tk.Scrollbar(transcript_frame, orient="vertical")
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


transcript_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=transcript_text.yview)

# Create frame for voice controls
voice_control_frame = tk.Frame(root)
voice_control_frame.pack(pady=10)

# Speed control
speed_label = tk.Label(voice_control_frame, text="Speed:")
speed_label.pack()
speed_var = tk.DoubleVar(value=1.0)
speed_slider = tk.Scale(voice_control_frame, from_=0.5, to=2.0, resolution=0.1,
                       orient=tk.HORIZONTAL, variable=speed_var, length=200)
speed_slider.pack()

# Pitch control
pitch_label = tk.Label(voice_control_frame, text="Pitch:")
pitch_label.pack()
pitch_var = tk.DoubleVar(value=1.0)
pitch_slider = tk.Scale(voice_control_frame, from_=0.5, to=2.0, resolution=0.1,
                       orient=tk.HORIZONTAL, variable=pitch_var, length=200)
pitch_slider.pack()

# Volume control
volume_label = tk.Label(voice_control_frame, text="Volume:")
volume_label.pack()
volume_var = tk.DoubleVar(value=1.0)
volume_slider = tk.Scale(voice_control_frame, from_=0.1, to=2.0, resolution=0.1,
                        orient=tk.HORIZONTAL, variable=volume_var, length=200)
volume_slider.pack()

# Preload the default language model (Hindi)
get_tts_model("Hindi")

# Start the queue processor
threading.Thread(target=process_speech_queue, daemon=True).start()

def reset_voice_controls():
    speed_var.set(1.0)
    pitch_var.set(1.0)
    volume_var.set(1.0)

reset_button = tk.Button(voice_control_frame, text="Reset Voice Settings", command=reset_voice_controls)
reset_button.pack(pady=5)

# Run the application
root.mainloop()