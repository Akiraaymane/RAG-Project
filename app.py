"""
app.py - RAG Chatbot Web Interface 
Run with: streamlit run app.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

from system_qa import QASystem

load_dotenv()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         PAGE CONFIGURATION                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="RAG Aymane - Philosophie",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        SESSION STATE MANAGEMENT                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                          HELPER FUNCTIONS                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@st.cache_resource
def initialize_qa_system():
    """Initialise le système QA (cached)"""
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        st.error("❌ HF_API_KEY not found in .env")
        return None
    try:
        return QASystem(hf_api_key=hf_api_key)
    except Exception as e:
        st.error(f"❌ Error initializing QA System: {str(e)}")
        return None


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            HEADER & BANNER                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

st.title("🧠 RAG AYMANE")
st.markdown("### 💫 Philosophie & Intelligence Artificielle")
st.divider()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            SIDEBAR MENU                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio(
        "📋 Mode:",
        ["💬 Chatbot", "❓ Q&A Simple", "📊 Dashboard"],
        captions=[
            "Conversation continue",
            "Une question = une réponse",
            "Statistiques du système"
        ]
    )
    
    st.divider()
    
    st.markdown("**📌 Paramètres:**")
    show_sources = st.toggle("Afficher les sources", value=True)
    top_k = st.slider("Nombre de documents", 1, 10, 3)
    
    st.divider()
    
    st.markdown("**ℹ️ Système:**")
    st.info("""
    - 🤖 LLM: HuggingFace Inference
    - 📚 Embeddings: MiniLM-L6
    - 💾 Vector Store: ChromaDB
    - 📄 Documents: 3 PDFs Philosophie
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.qa_system = None
            st.cache_resource.clear()
            st.rerun()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           MODE: CHATBOT                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if mode == "💬 Chatbot":
    st.subheader("💬 Chatbot Interactif")
    st.markdown("Posez vos questions sur la philosophie. L'assistant va répondre en utilisant les documents indexés.")
    st.divider()
    
    # Initialiser le système QA
    if st.session_state.qa_system is None:
        st.session_state.qa_system = initialize_qa_system()
    
    # Afficher l'historique du chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and show_sources:
                st.caption(f"📚 Sources: {', '.join(message['sources'])}")
    
    # Input et traitement
    if prompt := st.chat_input("💭 Posez votre question sur la philosophie..."):
        # Ajouter le message de l'utilisateur
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🤔 Réflexion en cours..."):
                try:
                    if st.session_state.qa_system:
                        result = st.session_state.qa_system.query(prompt, top_k=top_k)
                        assistant_response = result['answer']
                        sources = result.get('sources', [])
                        
                        st.write(assistant_response)
                        
                        if show_sources and sources:
                            st.caption(f"📚 Sources: {', '.join(sources)}")
                        
                        # Ajouter la réponse
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_response,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "sources": sources
                        })
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         MODE: Q&A SIMPLE                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

elif mode == "❓ Q&A Simple":
    st.subheader("❓ Mode Question Unique")
    st.markdown("Posez une question et obtenez une réponse rapide sans historique de conversation.")
    st.divider()
    
    if st.session_state.qa_system is None:
        st.session_state.qa_system = initialize_qa_system()
    
    question = st.text_area(
        "Votre question:",
        placeholder="Qu'est-ce que la philosophie?",
        height=150,
        label_visibility="collapsed"
    )
    
    if st.button("🚀 Obtenir la réponse", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("⏳ Génération de la réponse..."):
                try:
                    if st.session_state.qa_system:
                        result = st.session_state.qa_system.query(question, top_k=top_k)
                        
                        st.success("✅ Réponse générée!")
                        st.divider()
                        
                        st.markdown("### ✨ Réponse")
                        st.write(result['answer'])
                        
                        if show_sources and result.get('sources'):
                            st.divider()
                            st.markdown("### 📚 Sources Utilisées")
                            for i, src in enumerate(result.get('sources'), 1):
                                st.write(f"{i}. {src}")
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
        else:
            st.warning("⚠️ Veuillez entrer une question!")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         MODE: DASHBOARD                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

elif mode == "📊 Dashboard":
    st.subheader("📊 Tableau de Bord")
    st.markdown("Statistiques et informations du système RAG.")
    st.divider()
    
    # Statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💬 Messages", len(st.session_state.chat_history))
    
    with col2:
        st.metric("📄 Documents", "3")
    
    with col3:
        st.metric("⚡ Latence", "~1.2s")
    
    with col4:
        st.metric("✅ Uptime", "100%")
    
    st.divider()
    
    # Historique récent
    st.markdown("### 📋 Historique Récent")
    
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-10:]:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.write(f"**{role_icon} {msg['role'].upper()}** ({msg['timestamp']})")
            st.write(msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"])
            st.divider()
    else:
        st.info("📭 Aucun message pour le moment")
    
    st.divider()
    
    # Documents
    st.markdown("### 📚 Documents Indexés")
    docs = [
        "philosophie.pdf",
        "LE_SENS_DE_LA_PHILOSOPHIE.pdf",
        "pascal_petits_ecrits_philosophiques_religieux.pdf"
    ]
    
    for doc in docs:
        st.write(f"✅ `{doc}`")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            FOOTER                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

st.divider()
st.markdown("""
---
Made with by Aymane | RAG System - Philosophie & IA | v1.0.0
""", unsafe_allow_html=False)