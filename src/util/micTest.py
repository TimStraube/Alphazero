import pyaudio
import numpy as np
import time

# Initialize PyAudio
p = pyaudio.PyAudio()

# Print available devices
print("\nAVAILABLE AUDIO DEVICES:")
print("-----------------------")
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    print(f"Device {i}: {dev_info['name']}")
    print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
    print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")

# Try all devices with input channels
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0:
        try:
            print(f"\nTesting device {i}: {dev_info['name']}")
            print("Speak into the microphone. Monitoring for 10 seconds...")
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=min(2, dev_info['maxInputChannels']),  # Use 1 or 2 channels
                rate=int(dev_info['defaultSampleRate']),
                input=True,
                input_device_index=i,
                frames_per_buffer=1024
            )
            
            # Monitor for 10 seconds
            start_time = time.time()
            max_volume = 0
            
            while (time.time() - start_time) < 10:
                data = stream.read(1024, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                max_volume = max(volume, max_volume)
                
                # Visual volume meter
                meter = "#" * int(volume / 100)
                print(f"\rVolume: {volume:.0f} {meter}", end="")
                time.sleep(0.1)
            
            print(f"\nMax volume detected: {max_volume:.0f}")
            if max_volume > 100:
                print("✅ Microphone is working!")
            else:
                print("❌ Low or no audio detected. Microphone might not be working.")
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"\nError testing device {i}: {e}")
            continue

# Clean up
p.terminate()