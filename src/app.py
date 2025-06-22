import streamlit as st
import os
from emotion_converter import AudioEmotionConverter

def main():
    st.title("Audio Emotion Converter")
    st.write("Upload an audio file. The model and scaler will be loaded from the server.")

    # Hardcoded model and scaler paths (update these as needed)
    MODEL_PATH = "../G_neu2sad_final.pth"
    SCALER_PATH = None  # Set to path if you have a scaler

    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    # File uploader for audio files only
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

    # Emotion selection
    emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    target_emotion = st.selectbox("Select Target Emotion", emotions)

    if st.button("Convert Emotion"):
        if audio_file is not None:
            input_audio_path = os.path.join("temp", audio_file.name)
            with open(input_audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            scaler_path = SCALER_PATH if SCALER_PATH and os.path.exists(SCALER_PATH) else None
            try:
                converter = AudioEmotionConverter(MODEL_PATH, scaler_path)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return

            output_audio_path = os.path.join("temp", f"{audio_file.name.split('.')[0]}_{target_emotion}_converted.wav")
            try:
                # Use the correct method name from your class!
                if hasattr(converter, "process_audio"):
                    converter.process_audio(input_audio_path, target_emotion, output_audio_path)
                elif hasattr(converter, "convert_emotion_style"):
                    converter.convert_emotion_style(input_audio_path, target_emotion, output_audio_path)
                else:
                    st.error("No suitable method found in AudioEmotionConverter.")
                    return

                st.success("Audio conversion successful!")
                st.audio(output_audio_path, format='audio/wav')
                with open(output_audio_path, "rb") as f:
                    st.download_button("Download Converted Audio", f, file_name=os.path.basename(output_audio_path))
            except Exception as e:
                st.error(f"Error during conversion: {e}")
        else:
            st.error("Please upload an audio file.")

if __name__ == "__main__":
    main()