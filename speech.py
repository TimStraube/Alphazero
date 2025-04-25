"""
author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

import vosk
import pyaudio
import json
import sys
import time
import subprocess

try:
    # First, set the Sennheiser as default audio source using PulseAudio
    try:
        subprocess.run(["pactl", "set-default-source", 
                       "alsa_input.usb-Sennheiser_Sennheiser_SC_1x5_USB_A002740205001522-00.analog-stereo.2"], 
                       check=True)
        print("Set Sennheiser as default audio source")
    except Exception as e:
        print(f"Warning: Could not set default audio source: {e}")
    
    # Change to English model
    speech_model_path = "speech/models/vosk-model-small-en-us-0.15"
    speech_model = vosk.Model(speech_model_path)

    # Match sample rate with what's reported for your Sennheiser (32000 Hz)
    sample_rate = 32000
    rec = vosk.KaldiRecognizer(speech_model, sample_rate)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Print all available devices and their properties
    print("\nAVAILABLE AUDIO DEVICES:")
    print("-----------------------")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev_info['name']}")
        print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
        print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")
    
    # Specifically find and use pulse device for improved compatibility
    pulse_device_index = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if "pulse" in dev_info['name'].lower():
            pulse_device_index = i
            print(f"Selected PulseAudio device {i}: {dev_info['name']}")
            break
    
    # If we can't find pulse, use any device with input channels
    if pulse_device_index is None:
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                pulse_device_index = i
                print(f"Selected alternative input device {i}: {dev_info['name']}")
                break
    
    if pulse_device_index is None:
        print("No suitable input device found!")
        sys.exit(1)
    
    # Open the audio stream with pulse and use correct sample rate
    print("Opening audio stream...")
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,  # Your Sennheiser reports stereo (2 channels)
        rate=sample_rate,  # Using the 32000 Hz sample rate of your Sennheiser
        input=True,
        input_device_index=pulse_device_index,
        frames_per_buffer=4096
    )

    output_file_path = "recognized_text.txt"

    # Open a text file in write mode
    with open(output_file_path, "w") as output_file:
        print("Listening for speech. Say 'Terminate' to stop.")
        
        # Add visual feedback
        counter = 0
        
        # Start streaming and recognize speech
        while True:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                
                # Visual indicator that we're receiving audio
                if counter % 20 == 0:
                    print(".", end="", flush=True)
                counter += 1
                
                # Process audio data
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    recognized_text = result.get('text', '')
                    
                    if recognized_text:
                        # Clear the line of dots
                        print("\r" + " " * 50 + "\r", end="")
                        
                        # Display and save recognized text
                        print(f"Recognized: {recognized_text}")
                        output_file.write(recognized_text + "\n")
                        
                        # Check for termination keyword
                        if "terminate" in recognized_text.lower():
                            print("Termination keyword detected. Stopping...")
                            break
                        
                # Show partial results
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get('partial', '')
                if partial_text:
                    # Clear the line of dots
                    print("\r" + " " * 50 + "\r", end="")
                    print(f"Partial: {partial_text}", end="\r", flush=True)
                    
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                print(f"\nError reading audio: {e}")
                time.sleep(0.1)
                continue

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("\nSpeech recognition completed.")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)