import streamlit as st
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint

# Function to record audio
def record_audio(duration=10, sample_rate=44100, device=None):
    st.info("Recording audio... Please speak.")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16, device=device)
        sd.wait()
        st.info("Audio recording complete.")
        return audio, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None, None

# Function to transcribe audio to text
def transcribe_audio(audio, sample_rate):
    recognizer = sr.Recognizer()
    try:
        audio_data = sr.AudioData(audio.flatten(), sample_rate=sample_rate, sample_width=2)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Could not transcribe audio. Please try again with clearer audio.")
        return ""
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# Hugging Face API key and model configuration
hg_key = "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"

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    top_k=30,
    temperature=0.1,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hg_key
)

# Load text data
try:
    loader = TextLoader("doc_rag_copy.txt")
    full_text = loader.load()
except Exception as e:
    st.error(f"Error loading file: {e}")
    full_text = []

# Ensure text data is loaded
if not full_text:
    st.error("Failed to load text from the file.")
else:
    # Convert loaded text to a single string
    str_list = [str(num) for num in full_text]
    single_string = " ".join(str_list)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(single_string)

    # Create embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever()

    # Define template for chatbot
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}

    Suggest some funding schemes.

    """

    # Define RAG chatbot function
    def chat_with_rag(message):
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template(template)
        model = llm
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return chain.invoke(message)

# Define Streamlit UI
def main():
    st.set_page_config(page_title="WEP GenAI Chatbot", page_icon="🤖")

    st.title('WEP GenAI Chatbot')

    # Session state to manage recording and messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ''
    if 'audio' not in st.session_state:
        st.session_state.audio = None
    if 'sample_rate' not in st.session_state:
        st.session_state.sample_rate = 44100

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    # Audio input button (Hold to Talk)
    if st.button("🎤 ", help="Click for audio input",key="start_recording"):
        st.session_state.is_recording = True
        st.session_state.audio, st.session_state.sample_rate = record_audio(duration=5)  # You can adjust the duration as needed
        st.session_state.is_recording = False
        st.experimental_rerun()

    # Transcribe audio if available
    if st.session_state.audio is not None:
        transcribed_text = transcribe_audio(st.session_state.audio, st.session_state.sample_rate)
        if transcribed_text:
            st.session_state.transcribed_text = transcribed_text
            st.session_state.audio = None  # Reset audio state
            st.experimental_rerun()

    # User input
    if st.session_state.transcribed_text:
        st.session_state.messages.append({"role": "user", "text": st.session_state.transcribed_text})
        bot_response = chat_with_rag(st.session_state.transcribed_text)
        st.session_state.messages.append({"role": "bot", "text": bot_response})
        st.session_state.transcribed_text = ''  # Clear transcribed text after sending
        st.experimental_rerun()

    # Chat input box
    if transcribed_text := st.chat_input("Enter your query"):
        st.session_state.messages.append({"role": "user", "text": transcribed_text})
        bot_response = chat_with_rag(transcribed_text)
        st.session_state.messages.append({"role": "bot", "text": bot_response})
        st.experimental_rerun()

if __name__ == '__main__':
    main()
