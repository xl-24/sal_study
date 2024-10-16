#-----------------------------------------------------------------------------
# Test audio data processing based on hugging face code at 
# https://huggingface.co/learn/audio-course/chapter5/asr_models
# This is also a place for me to learn python
# XL: 10/6/2024
#-----------------------------------------------------------------------------
import torch
from datasets import load_dataset
from IPython.display import Audio
import matplotlib.pyplot as plt
from transformers import pipeline

# XL: load dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
print(dataset)

# XL: show dataset
sample = dataset[0]
data_out = sample["audio"]["array"]
rate_out = sample["audio"]["sampling_rate"]

print(data_out)
print(rate_out)

# plt.plot(data_out)
# plt.show()



# #print(sample)
# print(sample["text"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# pipe = pipeline(
#     "automatic-speech-recognition", model="openai/whisper-base", device=device
# )

# out_text = pipe(sample["audio"], max_new_tokens=256)
# print("---final output---\n")
# print(out_text['text'])