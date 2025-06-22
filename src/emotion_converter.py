import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import noisereduce as nr
import subprocess
import os


class EmotionClassifier(nn.Module):
    """CNN model for emotion classification"""
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.model(x)


class AudioEmotionConverter:
    def __init__(self, model_path, scaler_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_map = {
            0: 'ANG',
            1: 'DIS',
            2: 'FEA',
            3: 'HAP',
            4: 'NEU',
            5: 'SAD'
        }
        self.reverse_emotion_map = {v: k for k, v in self.emotion_map.items()}
        self.model = EmotionClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def extract_features(self, audio_path, sr=22050):
        try:
            audio, _ = librosa.load(audio_path, sr=sr, duration=3.0)
            target_length = sr * 3
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=128, 
                hop_length=512, 
                n_fft=2048
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
            return mel_spec_db
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_emotion(self, audio_path):
        features = self.extract_features(audio_path)
        if features is None:
            return None, None
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_emotion = self.emotion_map[predicted_class]
        confidence_scores = {self.emotion_map[i]: prob.item() for i, prob in enumerate(probabilities)}
        return predicted_emotion, confidence_scores
    
    def convert_emotion_style(self, audio_path, target_emotion, output_path):
        try:
            audio, sr = librosa.load(audio_path, sr=22050)
            if target_emotion == 'HAP':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                audio = librosa.effects.time_stretch(audio, rate=1.1)
            elif target_emotion == 'SAD':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-3)
                audio = librosa.effects.time_stretch(audio, rate=0.9)
            elif target_emotion == 'ANG':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
                audio *= 1.2
            elif target_emotion == 'FEA':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=3)
                t = np.linspace(0, len(audio) / sr, len(audio))
                tremolo = 1 + 0.3 * np.sin(2 * np.pi * 6.0 * t)
                audio *= tremolo
            elif target_emotion == 'DIS':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)
            audio = audio / np.max(np.abs(audio))
            noise_clip = audio[:int(sr * 0.5)]
            audio_denoised = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=1.0, stationary=False)
            sf.write(output_path, audio_denoised, sr)
            print(f"Converted and denoised audio saved to: {output_path}")
        except Exception as e:
            print(f"Error converting audio: {e}")

    def process_audio(self, input_path, target_emotion, output_path=None):
        print(f"Processing audio: {input_path}")
        current_emotion, confidence_scores = self.predict_emotion(input_path)
        if current_emotion is None:
            print("Failed to predict emotion")
            return
        print(f"Current emotion: {current_emotion}")
        print("Confidence scores:")
        for emotion, score in confidence_scores.items():
            print(f"  {emotion}: {score:.3f}")
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_{target_emotion}_converted{input_file.suffix}"
        print(f"Converting to: {target_emotion}")
        self.convert_emotion_style(input_path, target_emotion, output_path)
        print("Verifying conversion...")
        new_emotion, new_confidence = self.predict_emotion(output_path)
        print(f"Converted emotion: {new_emotion}")
        print("New confidence scores:")
        for emotion, score in new_confidence.items():
            print(f"  {emotion}: {score:.3f}")