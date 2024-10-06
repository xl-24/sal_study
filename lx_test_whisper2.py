#------------------------------------------------------------------------------
# This one is wokring
# example of decoding the ATC data
# XL: 10/5/2024
#------------------------------------------------------------------------------ 

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the processor and the model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load audio file and preprocess
audio_path = "/users/xili/Documents/GitHub/sal_study/dca_f2_1.mp3"
audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz audio

# Preprocess the audio
inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

# Generate transcription (you can adjust decoding parameters)
with torch.no_grad():
    predicted_ids = model.generate(inputs["input_features"])

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
