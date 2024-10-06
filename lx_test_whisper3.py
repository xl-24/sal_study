from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf
import sounddevice as sd
import librosa

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load the audio (e.g., .wav file)
#audio, sr = sf.read("/users/xili/Downloads/test_memo.wav")
#print(sr)
audio_path = "/users/xili/Documents/GitHub/sal_study/xl_voice_recording.mp3"
audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz audio
# Play the sound
sd.play(audio, samplerate=sr)
# Wait until sound has finished playing
sd.wait()



# Preprocess the audio
input_features = processor(audio, return_tensors="pt", sampling_rate=16000).input_features

# Generate the transcription in a specific language (e.g., French)
# Use the forced_decoder_ids to specify the language
forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")

# Perform the transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)