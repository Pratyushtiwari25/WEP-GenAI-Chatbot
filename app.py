import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint

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
    loader = TextLoader("doc_rag.txt")
    full_text = loader.load()
except Exception as e:
    print(f"Exception: {e}")
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

    # Generalize the template for chatbot
    template = """Answer the following question based on the context provided:

    {context}

    Question: {question}
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
        st.title('RAG Chatbot')

        user_input = st.text_input('You:', '')
        if st.button('Send'):
            if not user_input.strip():
                st.warning("Please enter a question.")
            else:
                bot_response = chat_with_rag(user_input)
                st.text_area('Bot:', value=bot_response, height=200, max_chars=None)

    if __name__ == '__main__':
        main()
