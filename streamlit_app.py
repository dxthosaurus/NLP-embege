import streamlit as st
import pandas as pd
import re
import joblib
import numpy as np

st.set_page_config(page_title="ABSA & NER Makan Bergizi Gratis", layout="wide")
st.title("🍽️ Analisis Sentimen Multi-Model & Ekstraksi Entitas")
st.markdown("Aplikasi NLP membandingkan prediksi 4 algoritma berbeda terkait Program Makan Bergizi Gratis.")

# ==========================================
# 1. LOAD 4 MODEL TERLATIH
# ==========================================
@st.cache_resource
def load_all_models():
    vec = joblib.load('tfidf_vectorizer.joblib')
    models_dict = {
        "SVM (Linear)": joblib.load('svm_model.joblib'),
        "Random Forest": joblib.load('random_forest_model.joblib'),
        "Naive Bayes": joblib.load('naive_bayes_model.joblib'),
        "Logistic Reg": joblib.load('logistic_regression_model.joblib')
    }
    return vec, models_dict

try:
    vectorizer, loaded_models = load_all_models()
except Exception as e:
    st.error("⚠️ Gagal memuat model. Pastikan kamu sudah menjalankan script export_models.py")
    st.stop()

# ==========================================
# 2. KAMUS NER & FUNGSI BIO TAGGING
# ==========================================
entities_dict = {
    'ASPECT': ['anggaran', 'dana', 'uang', 'pajak', 'korupsi', 'triliun', 'biaya',
               'menu', 'susu', 'makan', 'gizi', 'nasi', 'lauk', 'sayur', 'telur',
               'distribusi', 'merata', 'infrastruktur', 'fasilitas'],
    'PER': ['prabowo', 'gibran', 'gema', 'menteri', 'presiden', 'anies'],
    'ORG': ['pemerintah', 'dpr', 'sekolah', 'polisi', 'tentara', 'posyandu', 'umkm', 'sppg', 'tni', 'polri'],
    'LOC': ['papua', 'desa', 'daerah', 'jakarta', 'indonesia', 'kampung']
}

def detect_aspect_and_ner(text):
    text_lower = str(text).lower()
    detected_aspect = 'Umum'
    for aspect, keywords in entities_dict.items():
        if aspect == 'ASPECT' and any(keyword in text_lower for keyword in keywords):
            detected_aspect = 'Menu, Anggaran, atau Distribusi'
            break
            
    tokens = str(text).split()
    bio_tags = []
    prev_ent = None
    
    for token in tokens:
        clean_token = re.sub(r'[^\w\s]', '', token.lower())
        assigned_tag = "O"
        current_ent = None
        for ent_label, keywords in entities_dict.items():
            if clean_token in keywords:
                current_ent = ent_label
                break
        if current_ent:
            if prev_ent == current_ent:
                assigned_tag = f"I-{current_ent}"
            else:
                assigned_tag = f"B-{current_ent}"
        bio_tags.append((token, assigned_tag))
        prev_ent = current_ent
        
    return detected_aspect, bio_tags

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
user_input = st.text_area("✍️ Masukkan ulasan atau opini masyarakat:", height=120)

if st.button("🔍 Analisis dengan 4 Model", type="primary"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        with st.spinner("Mengekstrak fitur dan menjalankan model..."):
            aspek, bio_result = detect_aspect_and_ner(user_input)
            
            # Persiapan Teks untuk Prediksi
            feature_text = f"{aspek} {user_input}"
            input_vector = vectorizer.transform([feature_text])
            
            st.divider()
            
            # --- BAGIAN 1: ASPEK & NER ---
            st.subheader("1. Ekstraksi Aspek & Entitas Bernama (NER)")
            st.info(f"**Aspek Dominan:** {aspek}")
            
            html_ner = "<div style='line-height: 2.5; font-size: 1.1rem; margin-bottom: 20px;'>"
            for token, tag in bio_result:
                if tag == 'O':
                    html_ner += f"<span>{token} </span>"
                else:
                    tag_color = "#FFD700" if "ASPECT" in tag else "#87CEFA" if "ORG" in tag else "#98FB98" if "PER" in tag else "#FFA07A"
                    html_ner += f"<span style='background-color:{tag_color}; padding:0.2rem 0.6rem; border-radius:0.4rem; margin:0 0.2rem; color:#1a1a1a; font-weight: 500; border: 1px solid rgba(0,0,0,0.1);'>{token} <sub style='font-size:0.6em; opacity: 0.7;'>{tag}</sub></span> "
            html_ner += "</div>"
            st.markdown(html_ner, unsafe_allow_html=True)
            
            # --- BAGIAN 2: KOMPARASI SENTIMEN (3 TERATAS PER MODEL) ---
            st.subheader("2. Komparasi Prediksi Sentimen (Kepercayaan Model)")
            
            # Membuat 4 kolom untuk 4 model
            cols = st.columns(4)
            
            # Mengambil urutan kelas dari model (biasanya ['negative', 'neutral', 'positive'])
            classes = loaded_models["SVM (Linear)"].classes_ 
            
            for idx, (model_name, model) in enumerate(loaded_models.items()):
                # Dapatkan probabilitas untuk ke-3 kelas
                probabilities = model.predict_proba(input_vector)[0]
                
                # Menggabungkan nama kelas dan probabilitas, lalu diurutkan dari terbesar ke terkecil
                class_probs = list(zip(classes, probabilities))
                class_probs.sort(key=lambda x: x[1], reverse=True)
                
                pred_label = class_probs[0][0]
                
                # Atur UI tiap kolom
                with cols[idx]:
                    st.markdown(f"**{model_name}**")
                    
                    # Tampilkan Label Final
                    if pred_label == 'positive':
                        st.success("✅ POSITIVE")
                    elif pred_label == 'negative':
                        st.error("❌ NEGATIVE")
                    else:
                        st.warning("➖ NEUTRAL")
                    
                    # Tampilkan 3 hasil probabilitas teratas (semua kelas)
                    st.markdown("<small>Distribusi Probabilitas:</small>", unsafe_allow_html=True)
                    for label, prob in class_probs:
                        pct = prob * 100
                        # Warna bar menyesuaikan label
                        bar_color = "green" if label=="positive" else "red" if label=="negative" else "gray"
                        st.markdown(f"""
                        <div style="font-size:0.8rem; margin-top:5px;">
                            {label.upper()} ({pct:.1f}%)
                            <div style="width:100%; background-color:#e0e0e0; border-radius:3px; height:6px; margin-top:2px;">
                                <div style="width:{pct}%; background-color:{bar_color}; height:100%; border-radius:3px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
