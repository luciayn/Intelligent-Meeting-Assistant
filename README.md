# Intelligent-Meeting-Assistant

## Overview
The Intelligent Meeting Assistant is a web application designed to transcribe, summarize, and analyze video meetings. It leverages advanced machine learning models for object detection and speech recognition to provide real-time insights and summaries.

## Features
- **Real-time Transcription**: Transcribes audio from video meetings in real-time.
- **Object Detection**: Detects objects in the video to trigger specific actions.
- **Summarization**: Generates summaries of the transcriptions.
- **Key Words Extraction**: Identifies and lists key words from the transcription.
- **Ideas Section**: Displays predefined ideas for improving meetings.

## Project Structure

### Files
- **index.html**: The main HTML file that structures the web application layout.
- **index.js**: Contains the main JavaScript logic for handling video processing, audio recording, and interaction with web workers.
- **recorder.worklet.js**: An AudioWorklet script for processing audio data in real-time.
- **whisper.worker.js**: A Web Worker script for handling speech recognition using the Whisper model.
- **style.css**: The CSS file for styling the web application.
- **README.md**: This file, providing an overview and documentation for the project.

## Getting Started
### Prerequisites
- A modern web browser that supports Web Workers and AudioWorklets.
- Internet connection to load external libraries from CDN.

### Running the Project
1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Open `index.html` in your web browser.

## Usage
- Open the web application in your browser.
- Play the video to start transcription and object detection.
- View the transcription, key words, and summary sections for real-time updates.

## Dependencies
- [TensorFlow.js](https://cdn.jsdelivr.net/npm/@tensorflow/tfjs)
- [TensorFlow.js Vis](https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis)
- [Hugging Face Transformers](https://cdn.jsdelivr.net/npm/@huggingface/transformers)
- [Xenova Transformers](https://cdn.jsdelivr.net/npm/@xenova/transformers)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.