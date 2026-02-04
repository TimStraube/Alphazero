import pyaudio
import vosk
import json

def listen_and_recognize(model_path, device_index=None):
    """Record audio and perform real-time speech recognition."""
    p = pyaudio.PyAudio()
    
    # Find suitable device
    if device_index is None:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                device_index = i
                print(f"Using device {i}: {info['name']}")
                break
    
    # Load Vosk model
    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, 16000)
    
    # Configure stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=4096
    )
    
    print("Listening for speech...")
    
    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            
            # Process with Vosk
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recognized_text = result.get('text', '')
                
                if recognized_text:
                    print(f"\nRecognized: {recognized_text}")
                    
                    # Stop on "terminate" or other trigger word
                    if "terminate" in recognized_text.lower():
                        print("Termination command detected. Stopping...")
                        break
            
            # Show partial results
            partial = json.loads(rec.PartialResult())
            partial_text = partial.get('partial', '')
            if partial_text:
                print(f"\rPartial: {partial_text}", end="")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    # Close resources
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    model_path = "../speech/models/vosk-model-small-en-us-0.15"
    listen_and_recognize(model_path) 