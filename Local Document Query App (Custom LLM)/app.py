import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import PyPDF2
import google.generativeai as genai
import openai
import requests
from anthropic import Anthropic
import uuid



# Configuration
QDRANT_URL = "http://localhost:6333"  # Fixed to local Qdrant
vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(QDRANT_URL)

# Initialize Qdrant collection
def init_qdrant():
    qdrant.recreate_collection(
        collection_name="documents",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)  # 384 is the size of all-MiniLM-L6-v2 embeddings
    )

# Extract text from PDF or text file
def extract_text(file):
    if file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    return ""

# Vectorize and upload to Qdrant
def upload_to_qdrant(file_id, text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Simple chunking
    vectors = vectorizer.encode(chunks)
    points = [
        models.PointStruct(
            id=i,  # Use integer index as ID
            vector=vector.tolist(),
            payload={"text": chunk, "file_id": file_id}
        )
        for i, (vector, chunk) in enumerate(zip(vectors, chunks))
    ]
    qdrant.upsert(collection_name="documents", points=points)

# Query Qdrant and get Gemini response
def query_llm(query, llm_model, api_key):
    query_vector = vectorizer.encode(query).tolist()
    search_result = qdrant.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=3  # Get top 3 relevant chunks
    )
    context = "\n".join([result.payload["text"] for result in search_result])
    
    try:
        if llm_model.startswith("gemini"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(llm_model)
            prompt = f"{query}\n\nContext: {context}"
            response = model.generate_content(prompt)
            return response.text

        elif llm_model.startswith("gpt"):
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": f"{query}\n\nContext: {context}"}
                ]
            )
            return response.choices[0].message["content"]

        elif llm_model.startswith("mistral"):
            # Mistral API endpoint (replace with actual endpoint if different)
            url = "https://api.mixtral.ai/v1/chat/completions"  # Hypothetical, check Mistral docs
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": f"{query}\n\nContext: {context}"}
                ]
            }
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        elif llm_model.startswith("grok"):
            # Grok API endpoint (via xAI)
            url = "https://api.xai.com/v1/chat/completions"  # Hypothetical, check xAI docs
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": f"{query}\n\nContext: {context}"}
                ]
            }
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        elif llm_model.startswith("claude"):
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=llm_model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": f"{query}\n\nContext: {context}"}
                ]
            )
            return response.content[0].text

        else:
            return "Unsupported LLM model selected."
    except Exception as e:
        return f"Error with LLM: {str(e)}"

# Streamlit UI
def main():
    st.title("Local Document Query App (Custom LLM)")

    # Initialize Qdrant on first run
    if "qdrant_initialized" not in st.session_state:
        init_qdrant()
        st.session_state.qdrant_initialized = True
        
    # LLM selection and API key input
    llm_options = [
        "gemini-1.5-flash",
        "gemini-2.5-pro-exp-03-25",
        "gpt-3.5-turbo",
        "gpt-4" 
        "mistral-7b-instruct",  
        "mixtral-8x7b-instruct",  
        "grok-3"  
        "claude-3.5-sonnet"
    ]
    selected_llm = st.selectbox("Select LLM Model", llm_options, index=1)  # Default to gemini-2.5-pro-exp-03-25
    api_key = st.text_input(f"Enter API Key for {selected_llm}", type="password", value="")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])
    if uploaded_file:
        file_id = uploaded_file.name
        text = extract_text(uploaded_file)
        if text:
            upload_to_qdrant(file_id, text)
            st.success(f"Uploaded {file_id} to vector database!")

    # Query input
    query = st.text_input("Ask a question about your document:")
    if st.button("Submit") and query:
        if not api_key:
            st.error("Please provide an API key for the selected LLM.")
        else:
            with st.spinner(f"Generating response using {selected_llm}..."):
                response = query_llm(query, selected_llm, api_key)
                st.write("**Response:**")
                st.write(response)

if __name__ == "__main__":
    main()
    
