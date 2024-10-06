#------------------------------------------------------------------------------
# example of decoding the my voice data
# XL: 10/5/2024
#------------------------------------------------------------------------------ 

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import librosa
import sounddevice as sd

# Load model and processor
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()


    # Load and process audio
def load_audio(filepath):
    audio, sample_rate = torchaudio.load(filepath)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.size(0) > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    return audio

if True:
    # This method is not working    
    audio = load_audio("/users/xili/Documents/GitHub/sal_study/xl_voice_recording.mp3")
else:
    # Load audio file and preprocess
    audio_path = "/users/xili/Documents/GitHub/sal_study/xl_voice_recording.mp3"
    audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz audio
    # Play the sound
    sd.play(audio, samplerate=sr)
    # Wait until sound has finished playing
    sd.wait()

# Preprocess and generate transcription
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(model.device)
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# Decode the predicted IDs to get the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])