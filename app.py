import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
import gdown
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

# Tentukan direktori untuk menyimpan data FAISS
faiss_index_path = "./faiss_index_with_metadata"
metadata_path = f"{faiss_index_path}_metadata.npy"

# Periksa apakah GPU tersedia
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Inisialisasi model SentenceTransformer
sentence_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
sentence_model = sentence_model.to(device)

# Fungsi untuk memuat indeks FAISS dan metadata
def load_faiss_with_metadata(index_path, metadata_path):
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(index_path)
    metadata_array = np.load(metadata_path, allow_pickle=True)
    print("FAISS index and metadata loaded successfully.")
    return index, metadata_array

# Fungsi untuk mengunduh FAISS database jika belum ada
def download_faiss_db():
    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        print("Downloading FAISS database...")
        index_url = "https://drive.google.com/uc?id=1QhQfCjaoOOoLZaTCN37Jcuzt4hMJAzn_"
        metadata_url = "https://drive.google.com/uc?id=1-0dOlC8u2MPqS4W6KW5eo9VmL8RS4iHW"

        # Download FAISS index
        gdown.download(index_url, output=faiss_index_path, quiet=False)

        # Download metadata file
        gdown.download(metadata_url, output=metadata_path, quiet=False)

        print("Download complete.")

# Pastikan database tersedia
download_faiss_db()

# Muat kembali indeks FAISS dan metadata
index, metadata_array = load_faiss_with_metadata(faiss_index_path, metadata_path)

# Fungsi untuk menghasilkan jawaban menggunakan GenAI
def generate_answer_with_genai(query, context):
    genai.configure(api_key=os.getenv("API_KEY"))
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(f"Pertanyaan: {query}\nKonteks: {context}")
    return response.text

# Streamlit app
st.header("RAG Local Wisdom")

input_user = st.text_input("Masukan pertanyaan")
button_submit = st.button("Submit", type="primary")

if button_submit:
    if not input_user:
        st.warning("Harap masukkan pertanyaan terlebih dahulu.")
    else:
        # Encode pertanyaan
        query_embedding = sentence_model.encode(input_user, convert_to_numpy=True, device=device)
        query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS memerlukan bentuk (1, dim)

        # Cari top-k dokumen yang paling mirip
        k = 5
        distances, indices = index.search(query_embedding, k)

        # Ambil konteks dokumen terkait
        related_documents = []
        for idx in indices[0]:
            if idx < len(metadata_array):
                metadata_item = metadata_array[idx].split("|||")
                related_documents.append(metadata_item[1])

        # Gabungkan konteks untuk diberikan ke GenAI
        context = "\n".join(related_documents)

        # Dapatkan jawaban dari GenAI
        answer = generate_answer_with_genai(input_user, context)

        # Tampilkan hasil
        st.subheader("Jawaban:")
        st.write(answer)

        st.subheader("Dokumen terkait:")
        for doc in related_documents:
            st.write(f"- {doc}")