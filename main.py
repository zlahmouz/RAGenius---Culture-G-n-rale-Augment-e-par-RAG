import streamlit as st
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA Culture Générale",
    page_icon="🤖",
    layout="wide"
)

# Caching des modèles
@st.cache_resource
def load_models(model_name="google/flan-t5-large"):
    """Charge et met en cache les modèles."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return embedding_model, tokenizer, model

# Chargement direct du fichier
@st.cache_data
def load_documents(file_path="culture_generale.txt"):
    """Charge les documents depuis le fichier texte."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        documents = [line.strip() for line in data if line.strip()]
        return documents
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return None

def create_faiss_index(documents, embedding_model):
    """Crée un index FAISS à partir des documents."""
    with st.spinner("Création de l'index FAISS en cours..."):
        embeddings = embedding_model.encode(documents, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index, embeddings

def retrieve_passages(query, embedding_model, index, documents, top_k=3):
    """Récupère les passages les plus pertinents."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]], distances[0]

def generate_answer(context, question, tokenizer, model):
    """Génère une réponse basée sur le contexte et la question."""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interface Streamlit
st.title("🤖 Assistant IA Culture Générale")
st.markdown("""
Cette application utilise l'IA pour répondre à vos questions de culture générale en se basant sur une base de connaissances.
""")

# Initialisation des modèles
try:
    embedding_model, tokenizer, model = load_models()
    st.sidebar.success("✅ Modèles chargés avec succès!")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles: {str(e)}")
    st.stop()

# Chargement direct des documents
if 'documents' not in st.session_state:
    st.session_state.documents = load_documents()
    
if 'index' not in st.session_state and st.session_state.documents:
    st.session_state.index, _ = create_faiss_index(st.session_state.documents, embedding_model)

# Affichage des statistiques dans la barre latérale
st.sidebar.header("📊 Statistiques")
if st.session_state.documents:
    st.sidebar.metric("Documents chargés", len(st.session_state.documents))

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input("🤔 Posez votre question:", placeholder="Exemple: Qui a gagné la Coupe du Monde 2018?")
    
    if st.button("🔍 Rechercher", disabled=not st.session_state.documents):
        if not question:
            st.warning("⚠️ Veuillez poser une question!")
        else:
            with st.spinner("🤔 Recherche en cours..."):
                # Récupération des passages pertinents
                relevant_passages, scores = retrieve_passages(
                    question,
                    embedding_model,
                    st.session_state.index,
                    st.session_state.documents
                )
                
                # Génération de la réponse
                context = " ".join(relevant_passages)
                answer = generate_answer(context, question, tokenizer, model)
                
                # Affichage des résultats
                st.success("✨ Réponse générée!")
                st.markdown("### 📝 Réponse:")
                st.markdown(f"**{answer}**")
                
                # Affichage des sources
                with st.expander("📚 Sources utilisées"):
                    for passage, score in zip(relevant_passages, scores):
                        st.markdown(f"- 📖 {passage}")
                        st.progress(float(score))
                
                # Ajout à l'historique
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append(question)

with col2:
    st.markdown("### 📜 Historique des questions")
    if 'history' in st.session_state:
        for q in st.session_state.history:
            st.markdown(f"- {q}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Développé avec ❤️ | Utilise FAISS et Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)
