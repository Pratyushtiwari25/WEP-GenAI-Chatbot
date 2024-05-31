# chatbot

Readme file-

# RAG Chatbot for Government Schemes under WEP

This project is a Generative AI chatbot that uses Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) technology to provide information about various government schemes under the Women Entrepreneurship Platform (WEP).

## Table of Contents

- (#Introduction)
- (#Features)
- (#Installation)
- (#Usage)
- (#Dataset)
- (#Contributing)
- (#License)

## Introduction

This chatbot leverages state-of-the-art LLM from Hugging Face and FAISS for efficient text retrieval and generation. The chatbot is designed to assist users in finding relevant government schemes under the WEP by answering queries based on a provided dataset.

## Features

- **Interactive Chat Interface**: Allows users to interact with the chatbot via a Streamlit web interface.
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval mechanisms with text generation to provide accurate and relevant responses.
- **Hugging Face Integration**: Utilizes Hugging Face models for natural language processing tasks.

## Installation

To run this project, follow these steps:

1. **Clone the repository**:
    git clone https://github.com/your-username/rag-chatbot-wep.git
    cd rag-chatbot-wep

2. **Install the required dependencies**:
    pip install -r requirements.txt
  

3. **Add your Hugging Face API Key**:
    - Replace `hg_key` in the code with your Hugging Face API key.

## Usage

To start the chatbot, run the following command:
streamlit run app.py

This will launch a Streamlit web application. You can enter your questions about government schemes under WEP into the text input box and receive responses from the chatbot.

## Dataset

The dataset used by the chatbot contains information on various government schemes under the WEP. The data is loaded from a text file named `doc_rag.txt`. Ensure that this file is in the same directory as your code.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

