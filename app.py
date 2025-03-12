import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Streamlit page setup
st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Pinecone setup
INDEX_NAME = "lawdata-index"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    st.error(f"❌ Index '{INDEX_NAME}' not found.")
    st.stop()

# Initialize Pinecone index
index = pc.Index(INDEX_NAME)

# Load embedding & reranking models
embedding_model = SentenceTransformer("BAAI/bge-large-en")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

st.title("⚖️ LEGAL ASSISTANT")

st.markdown("This AI-powered legal assistant retrieves relevant legal documents and generates professional legal reports.")

# User Input
query = st.text_input("🔍 Enter your legal query:")

if st.button("Generate Report"):
    if not query:
        st.warning("⚠️ Please enter a legal question before generating a report.")
        st.stop()

    # Convert user query into embeddings
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    # Query Pinecone for similar embeddings
    try:
        search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    except Exception as e:
        st.error(f"❌ Pinecone query failed: {e}")
        st.stop()

    # Handle missing results
    if not search_results or "matches" not in search_results or not search_results["matches"]:
        st.warning("⚠️ No relevant case found. Please refine your query.")
        st.stop()

    # Extract retrieved case text & ensure they are from Pinecone
    retrieved_cases = []
    case_citations = []

    for match in search_results["matches"]:
        if "text" in match["metadata"]:
            case_text = match["metadata"]["text"]
            case_source = match["metadata"].get("source", "Unknown Case")

            retrieved_cases.append(f"{case_text}")  # ✅ No unnecessary formatting

            if case_source != "Unknown Case":
                case_citations.append(f"{case_source}")  # ✅ Case citations without brackets

    # Prevent hallucination: Stop if no valid retrieved cases
    if not retrieved_cases:
        st.warning("⚠️ No relevant case found. Please refine your query.")
        st.stop()

    # Combine retrieved cases (limit to 5 for better context)
    context_text = "\n\n".join(retrieved_cases[:5])

    # 🔥 STRICT LLM Prompt to prevent hallucination
    prompt = f"""
    Generate a **formal legal report** based only on the provided legal materials. 
    Ensure the report follows a **professional tone** and is formatted for legal use.

    **Report Structure:**  
    - **Introduction**: Overview of the case.  
    - **Facts of the Case**: Key facts and procedural history.  
    - **Legal Issues**: Relevant legal provisions and arguments.  
    - **Court’s Reasoning**: Judicial interpretation and key findings.  
    - **Final Ruling**: The court’s decision.  
    - **Citations**: Legal sources used.  

    **Important Rules:**  
    - **Do NOT fabricate case law, statutes, or legal principles.**  
    - **Ensure citations are included naturally in the report.**  
    - **Do NOT mention retrieval or missing data.**  

    📜 **Retrieved Legal Context:**  
    {context_text}

    **Legal Report:**  
    """

    # Query Together AI with strict constraints
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert in legal matters."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )

        response_data = response.json()
        answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not answer or "No relevant case found" in answer:
            st.warning("⚠️ No relevant legal case found. Please refine your query.")
            st.stop()

    except Exception as e:
        st.error(f"❌ AI query failed: {e}")
        st.stop()

    # Display results
    st.success("📜 **Legal Report Generated:**")
    st.markdown(answer, unsafe_allow_html=True)

    # Show referenced cases
    if case_citations:
        st.markdown("### 📌 **Referenced Cases:**")
        st.markdown(", ".join(set(case_citations)))

# Footer
st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit, Pinecone, and Llama-3.3-70B-Turbo on Together AI</p>", unsafe_allow_html=True)
