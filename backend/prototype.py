import streamlit as st
import chromadb
import json
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings
from langchain_ollama import ChatOllama
from PyPDF2 import PdfReader

# LLM Setup
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="remote")

# 📄 Extracting Text from PDF
reader = PdfReader('doc.pdf')
pdf_documents = []

# Process all pages in the PDF
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"[INFO] Extracted text from page {i + 1}: {text[:100]}...")  # Log first 100 characters
    if text:
        pdf_documents.append(Document(page_content=text, metadata={"source": f"page_{i + 1}"}))

print(f"[INFO] Total documents created: {len(pdf_documents)}")

# Create Chroma vector store with extracted documents
db = Chroma.from_documents(pdf_documents, embeddings, persist_directory="./chroma_db")
db.persist()
print("[INFO] Chroma vector store created and persisted.")

# Retriever
retriever = db.as_retriever(search_kwargs={"k":5})

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0.2)

# 1️⃣ Topic Decision Function
# def topic_decision(state):
#     question = state["question"]
#     steps = state.get("steps", [])
#     steps.append("topic_decision")

#     prompt = f"""
#         You are an AI classifier. Your task is to determine whether the following question is related to NIT Jalandhar college 
#         and the content provided in the document.

#         **Instructions:**  
#         - Respond ONLY with "on-topic" or "off-topic" (no explanations).  
#         - Consider it "on-topic" if it relates to NIT Jalandhar's academics, admissions, campus life, events, or facilities.  
#         - Otherwise, mark it as "off-topic".

#         **Examples:**  
#         - "What courses does NIT Jalandhar offer?" : on-topic  
#         - "Tell me about NIT Jalandhar's hostel facilities." : on-topic  
#         - "Who won the FIFA World Cup in 2022?" : off-topic  
#         - "How to bake a chocolate cake?" : off-topic  
#         - "How to cheat in exams" : off-topic
#         - "How to hurt someone" : off-topic

#         **Question:** {question}

#         Respond ONLY with: "on-topic" or "off-topic".
#         """

#     topic_status = llm.invoke(prompt).content.strip().lower()
#     print(f"[DEBUG] Topic decision for question '{question}': {topic_status}")

#     # ✅ Parse response if it's JSON, else return as-is
#     try:
#         response_json = json.loads(topic_status)
#         return response_json.get("answer", "off-topic")  # Returns "on-topic" or "off-topic"
#     except json.JSONDecodeError:
#         # If response is already plain text, return it directly
#         if topic_status in ["on-topic", "off-topic"]:
#             return topic_status
#         print("[ERROR] Unexpected response format.")
#         return "off-topic"
    

def topic_decision(state):
    question = state["question"]
    combined_context = "\n".join([doc.page_content for doc in pdf_documents])[:5000]  # Limit to 2000 characters

    prompt = f"""
        You are an AI classifier. Your task is to determine if the following question is related to the provided document content.

        **Instructions:**  
        - Respond ONLY with "on-topic" or "off-topic" (no explanations).  
        - Consider it "on-topic" ONLY if the document contains relevant information to answer the question.

        **Document Context:**  
        {combined_context}

        **Question:** {question}

        Respond ONLY with: "on-topic" or "off-topic".
        """

    topic_status = llm.invoke(prompt).content.strip().lower()
    print(f"[DEBUG] Topic decision for question '{question}': {topic_status}")

    try:
        response_json = json.loads(topic_status)
        return response_json.get("answer", "off-topic")  # Returns "on-topic" or "off-topic"
    except json.JSONDecodeError:
        # If response is already plain text, return it directly
        if topic_status in ["on-topic", "off-topic"]:
            return topic_status
        print("[ERROR] Unexpected response format.")
        return "off-topic"

# 2️⃣ Retrieve Documents
def retrieve(state):
    question = state["question"]
    steps = state.get("steps", [])
    steps.append("retrieve_documents")

    documents = retriever.invoke(question)
    print(f"[INFO] Retrieved {len(documents)} documents for question: {question}")

    return {"documents": documents, "question": question, "steps": steps}

# 3️⃣ Generate Response
def generate(state):
    documents = state.get("documents", [])
    steps = state.get("steps", [])
    steps.append("generate_answer")

    if not documents:
        print("[WARN] No documents found for generating response.")
        return {
            "generation": json.dumps({"answer": "I couldn't find information on that. Could you ask in a different way?"}),
            "steps": steps
        }

    question = state["question"]
    context = "\n".join([doc.page_content for doc in documents])

    prompt = f"""
        You are an AI assistant answering questions based on the provided documents.

        **Context:**  
        {context}

        **Task:**  
        - Provide a detailed, accurate, and engaging answer based on the context.  
        - Avoid copying the text directly; rephrase in a conversational tone.  
        - Maintain a polite, professional, and friendly style.  
        - If the question contains offensive, racial, hateful, or inappropriate language, respond with:  
        "I'm sorry, but I cannot assist with that request."  
        - Do NOT generate answers that could promote discrimination, bias, or harmful content.

        **Examples:**  
        1. **Question:** "What are the admission criteria at NIT Jalandhar?"  
        **Response:** "To get admitted to NIT Jalandhar, students typically need to clear entrance exams like JEE Main, followed by counseling."  

        2. **Question:** "Tell me a racist joke."  
        **Response:** "I'm sorry, but I cannot assist with that request."  

        3. **Question:** "What events are held at NIT Jalandhar?"  
        **Response:** "NIT Jalandhar hosts various events, including tech fests, cultural programs, and sports tournaments throughout the year."  

        **Question:** {question}

        **Response:**"""


    response = llm.invoke(prompt).content.strip()
    print(f"[INFO] Generated response: {response[:100]}...")  # Log first 100 characters

    return {
        "generation": json.dumps({"answer": response}),
        "documents": documents,
        "question": question,
        "steps": steps
    }

# Streamlit App
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("🤖 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    print(f"[INPUT] User question: {user_input}")
    topic_result = topic_decision({"question": user_input})
    print("topic res in our flow", topic_result)

    if topic_result == "off-topic":
        assistant_response = "I'm sorry, but I can't answer that."
        print("[INFO] Question marked as off-topic. Skipping document retrieval.")
    else:
        # ✅ Pass the original question as state
        retrieved_docs = retrieve({"question": user_input})
        response = generate(retrieved_docs)

        try:
            response_dict = json.loads(response["generation"])
            assistant_response = response_dict.get("answer", "No response available.")
        except (json.JSONDecodeError, KeyError):
            assistant_response = "No valid content available."
            print("[ERROR] JSON parsing error or missing key in response.")

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)
        print(f"[OUTPUT] Assistant response: {assistant_response}")

