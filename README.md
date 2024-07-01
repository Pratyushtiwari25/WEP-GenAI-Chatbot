# WEP GenAI Chatbot

WEP GenAI Chatbot is an interactive AI chatbot built using Streamlit that allows users to interact with it through both text and voice inputs. The chatbot leverages LangChain for text processing and Hugging Face's models for text generation. It also utilizes FAISS for efficient similarity search.

## Features

- **Audio Recording**: Records audio input from the user.
- **Speech Recognition**: Converts recorded audio to text using Google's speech recognition.
- **Text Processing**: Splits text into manageable chunks and creates embeddings.
- **Retrieval-Augmented Generation (RAG)**: Answers questions based on provided context using a retriever and a language model.
- **Streamlit UI**: A web interface to interact with the chatbot.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/wep-genai-chatbot.git
   cd wep-genai-chatbot
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Create a file named `doc_rag_copy.txt` in the root directory and add your text data to it.

## Usage

1. Set your Hugging Face API key in the code .

2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

3. Open the provided local URL in your web browser to interact with the chatbot.

## Screenshots

1. Chat Interface
<img width="925" alt="Screenshot 2024-07-01 at 6 20 10 PM" src="https://github.com/Pratyushtiwari25/chatbot/assets/125774489/2b9bf846-5938-4958-b559-0b7fa84adc9a">


2. Chat Chaining 
<img width="863" alt="Screenshot 2024-07-01 at 6 20 12 PM" src="https://github.com/Pratyushtiwari25/chatbot/assets/125774489/db07d9d5-e9d7-41b6-8939-ed7eaa7c805f">

## Code Overview

### `app.py`

- **Audio Recording**: The `record_audio` function captures audio from the user's microphone.
- **Speech Recognition**: The `transcribe_audio` function converts the recorded audio to text using Google's speech recognition API.
- **Text Processing**: Text is loaded from a file, split into chunks, and indexed using FAISS.
- **RAG Chatbot**: The `chat_with_rag` function generates responses based on the user's query and the provided context.
- **Streamlit UI**: The `main` function sets up the Streamlit interface, handling user inputs and displaying messages.

## Dependencies

- streamlit
- sounddevice
- numpy
- speech_recognition
- langchain
- FAISS
- HuggingFace

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://langchain.com/)

---

Feel free to customize this README file further according to your project's specifics and preferences.
