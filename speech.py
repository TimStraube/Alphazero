"""
author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

import vosk
import pyaudio
import json

speech_model_path = "speech/models/vosk-model-small-de-0.15"
speech_model = vosk.Model(speech_model_path)

rec = vosk.KaldiRecognizer(speech_model, 16000)

p = pyaudio.PyAudio()
stream = p.open(
    format = pyaudio.paInt16,
    channels = 1,
    rate = 16000,
    input = True,
    frames_per_buffer = 8192
)

output_file_path = "recognized_text.txt"

# Open a text file in write mode using a 'with' block
with open(output_file_path, "w") as output_file:
    print("Listening for speech. Say 'Terminate' to stop.")
    # Start streaming and recognize speech
    while True:
        data = stream.read(4096)
        #read in chunks of 4096 bytes
        if rec.AcceptWaveform(data):
            #accept waveform of input voice
            # Parse the JSON result and get the recognized text
            result = json.loads(rec.Result())
            recognized_text = result['text']
            
            # Write recognized text to the file
            output_file.write(recognized_text + "\n")
            print(recognized_text)
            
            # Check for the termination keyword
            if "terminate" in recognized_text.lower():
                print(
                    "Termination keyword detected. Stopping..."
                )
                break

stream.stop_stream()
stream.close()

# Terminate the PyAudio object
p.terminate()

# if you don't want to download the model, just mention "lang" argument 
# in vosk.Model() and it will download the right  model, here the language is 
# US-English
# model = vosk.Model(lang="en-us")