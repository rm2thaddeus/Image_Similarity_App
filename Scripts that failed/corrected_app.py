 import streamlit as st
import sqlite3
import faiss
import numpy as np
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model and processor
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# Connect to the SQLite database
@st.cache_resource
def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    return conn

# Load embeddings and file paths from the database
@st.cache_data
def load_embeddings(_conn):
    cursor = _conn.cursor()
    cursor.execute("SELECT image_path, embedding FROM Embeddings")
    data = cursor.fetchall()
    file_paths = []
    embeddings = []
    for file_path, embedding in data:
        file_paths.append(image_path)
                # Embeddings are stored as BLOBs
        embedding = np.frombuffer(embedding, dtype=np.float32)
        embeddings.append(embedding)
    embeddings = np.array(embeddings).astype('float32')
    return file_paths, embeddings

# Initialize FAISS index
@st.cache_resource
def create_faiss_index(embeddings):
    # Check if embeddings are loaded correctly
    if len(embeddings) == 0:
        st.error("No embeddings loaded. Please check the database.")
        return None

    # Ensure embeddings are a 2D array
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")  # Debugging line

    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        st.error("Embeddings have an unexpected shape. Expected a 2D array with non-zero dimensions.")
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

# Streamlit UI
def main():
    st.title("Image Vector Database Tester")

    # Sidebar for database path
    db_path = st.sidebar.text_input("Path to vector_database.db", value="vector_database.db")
    if not db_path:
        st.error("Please provide the path to your vector_database.db")
        return

    # Establish database connection
    conn = get_db_connection(db_path)
    file_paths, embeddings = load_embeddings(conn)
    index = create_faiss_index(embeddings)

    # Image uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and get embedding
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            image_embedding = image_features.cpu().numpy().astype('float32')
            # Normalize if necessary
            image_embedding /= np.linalg.norm(image_embedding, axis=1, keepdims=True)

        # Search in FAISS index
        k = 5  # Number of similar images to retrieve
        distances, indices = index.search(image_embedding, k)

        st.subheader("Top Similar Images:")
        cols = st.columns(k)
        for i in range(k):
            idx = indices[0][i]
            sim_image_path = file_paths[idx]
            try:
                sim_image = Image.open(sim_image_path)
                cols[i].image(sim_image, caption=f"Similarity: {distances[0][i]:.2f}", use_column_width=True)
            except Exception as e:
                cols[i].write(f"Error loading image: {sim_image_path}")

if __name__ == "__main__":
    main()
