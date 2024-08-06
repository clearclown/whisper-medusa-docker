# src/main.py
import os
import sys
import torch
import torchaudio
from dotenv import load_dotenv
from whisper_medusa import WhisperMedusaModel
from transformers import WhisperProcessor
import pdb

load_dotenv()

def setup_debugger():
    if os.getenv('DEBUG', 'False').lower() == 'true':
        def debug_exception(type, value, tb):
            pdb.post_mortem(tb)
        sys.excepthook = debug_exception
        print("Debugger is active")

def main():
    setup_debugger()

    model_name = os.getenv('MODEL_NAME')
    audio_path = os.getenv('AUDIO_PATH')
    language = os.getenv('LANGUAGE')
    SAMPLING_RATE = 16000

    print(f"Loading model: {model_name}")
    model = WhisperMedusaModel.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    print(f"Processing audio file: {audio_path}")
    input_speech, sr = torchaudio.load(audio_path)
    if input_speech.shape[0] > 1:  # If stereo, average the channels
        input_speech = input_speech.mean(dim=0, keepdim=True)

    if sr != SAMPLING_RATE:
        input_speech = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(input_speech)

    input_features = processor(input_speech.squeeze(), return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_features = input_features.to(device)
    model = model.to(device)

    print("Generating transcription...")
    model_output = model.generate(
        input_features,
        language=language,
    )
    predict_ids = model_output[0]
    pred = processor.decode(predict_ids, skip_special_tokens=True)
    print("Transcription:")
    print(pred)

if __name__ == "__main__":
    main()