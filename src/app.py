import streamlit as st
import os
from emotion_converter import AudioEmotionConverter

def main():
    # Inject custom CSS for a modern look
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
            color: #22223b;
        }
        .stButton > button {
            background-color: #4f8cff;
            color: white;
            border-radius: 8px;
            padding: 0.5em 2em;
            font-weight: 600;
            border: none;
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background-color: #2563eb;
        }
        .stFileUploader, .stSelectbox, .stTextInput, .stAudio {
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 1em;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #2563eb;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🔔 Info message at the top
    st.markdown(
        "<h4 style='color:#d97706;'>⚠️ Due to resource limitations, the app has been shifted to a Python-based Streamlit version for now.</h4>",
        unsafe_allow_html=True
    )

    st.title("Audio Emotion Converter")
    st.write("Upload an audio file. The model and scaler will be loaded from the server.")

    # Hardcoded model and scaler paths (update these as needed)
    MODEL_PATH = "G_neu2sad_final.pth"
    SCALER_PATH = None  # Set to path if you have a scaler

    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    # File uploader for audio files only
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

    # Only 'SAD' is enabled, others are disabled and marked as coming soon
    emotion_labels = ["SAD"]
    st.selectbox("Select Target Emotion", emotion_labels, index=0, help="Other emotions coming soon! Only SAD is available.")
    st.info("Other emotions will be added soon!")
    target_emotion = "SAD"

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
