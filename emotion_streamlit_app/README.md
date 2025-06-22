# Emotion Classification and Conversion Web App

This project is a Streamlit application that allows users to classify and convert emotions in audio files. It utilizes a Convolutional Neural Network (CNN) model for emotion classification and provides functionality to convert audio to a specified emotion style.

## Project Structure

```
emotion_streamlit_app
├── src
│   ├── emotion_converter.py  # Core functionality for emotion classification and conversion
│   └── app.py                # Entry point for the Streamlit application
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd emotion_streamlit_app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an audio file (WAV format) that you want to analyze.

4. Select the target emotion you want to convert the audio to from the dropdown menu.

5. Click the "Convert" button to process the audio.

6. The application will display the predicted current emotion and the converted audio file.

## Features

- Emotion classification using a trained CNN model.
- Conversion of audio to specified emotion styles (e.g., Happy, Sad, Angry).
- Denoising of audio using advanced techniques.
- User-friendly web interface for easy interaction.

## Dependencies

The project requires the following Python libraries:

- Streamlit
- PyTorch
- librosa
- soundfile
- numpy
- scikit-learn
- noisereduce

Make sure to install all dependencies listed in `requirements.txt`.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.