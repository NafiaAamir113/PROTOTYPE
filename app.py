import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Streamlit page setup
st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Initialize Pinecone
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

# Page Title
st.title("⚖️ LEGAL ASSISTANT")

st.markdown("This AI-powered legal assistant retrieves relevant legal documents and generates professional legal reports.")

# User Input
query = st.text_input("🔍 Enter your legal query:")

if st.button("Generate Report"):
    if not query:
        st.warning("⚠️ Please enter a legal question before generating a report.")
        st.stop()

    # **Extract Case Number for Exact Matching (Prevents Cross-Case Retrieval)**
    case_number = None
    if any(x in query for x in ["W.P. No.", "Crl.", "Civ.", "S.C.", "P.L.D."]):  # ✅ Covers various case types
        case_number = query.split()[-1].strip()  # Extracts last number in query

    # Convert user query into embeddings
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    # Query Pinecone for similar embeddings with a **minimum similarity threshold**
    try:
        search_results = index.query(vector=query_embedding, top_k=15, include_metadata=True)
    except Exception as e:
        st.error(f"❌ Pinecone query failed: {e}")
        st.stop()

    # Handle missing results
    if not search_results or "matches" not in search_results or not search_results["matches"]:
        st.warning("⚠️ No relevant case found. Please refine your query.")
        st.stop()

    # Extract and rerank retrieved case text
    retrieved_cases = []
    case_citations = []

    for match in search_results["matches"]:
        similarity_score = match.get("score", 0)  # Extract similarity score

        if similarity_score < 0.75:  # ✅ Only use high-confidence results
            continue

        if "text" in match["metadata"]:
            case_text = match["metadata"]["text"]
            case_source = match["metadata"].get("source", "Unknown Case")
            retrieved_case_number = match["metadata"].get("case_number", "")

            # ✅ **Enforce Exact Case Matching** (Prevents cross-document hallucination)
            if case_number and case_number not in retrieved_case_number:
                continue  # Skip irrelevant cases

            retrieved_cases.append(case_text)
            if case_source != "Unknown Case":
                case_citations.append(case_source)

    # **Prevent Hallucination: Stop if No Valid Retrieved Cases**
    if not retrieved_cases:
        st.warning(f"⚠️ No exact match found for case '{case_number}'. Please refine your query.")
        st.stop()

    # **Rerank results** before passing to LLM
    ranked_results = sorted(
        zip(retrieved_cases, reranker.predict([(query, case) for case in retrieved_cases])),
        key=lambda x: x[1], reverse=True
    )

    # Select top 5 cases
    context_text = "\n\n".join([r[0] for r in ranked_results[:5]])

    # **STRICT LLM Prompt to Prevent Hallucination**
    prompt = f"""
    You are a legal expert generating a report based strictly on the provided case law.  
    If there is insufficient retrieved data, respond with: "No relevant legal information found."

    **Legal Context (STRICTLY use this information ONLY):**  
    {context_text}

    **Generate a professional legal report based strictly on the above context.**
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
                "temperature": 0.0  # ✅ **Reduce randomness & hallucination**
            }
        )

        response_data = response.json()
        answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not answer or "No relevant legal information found" in answer:
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
