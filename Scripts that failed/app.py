import streamlit as st
import sqlite3
import faiss
import numpy as np
from PIL import Image
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
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# List all tables in the database
@st.cache_data
def list_tables(_conn):
    cursor = _conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

# Load embeddings and image paths from the specified table
@st.cache_data
def load_embeddings(_conn, table_name):
    cursor = _conn.cursor()
    try:
        cursor.execute(f"SELECT image_path, embedding FROM {table_name}")
    except sqlite3.OperationalError as e:
        st.error(f"Error accessing table `{table_name}`: {e}")
        return [], []
    data = cursor.fetchall()
    image_paths = []
    embeddings = []
    for image_path, embedding in data:
        image_paths.append(image_path)
        # Parse the BLOB to a NumPy array
        if isinstance(embedding, bytes):
            # Assuming embeddings are stored as float32 in BLOBs
            embedding = np.frombuffer(embedding, dtype=np.float32)
        else:
            st.warning(f"Unknown embedding format for {image_path}")
            continue
        embeddings.append(embedding)
    if not embeddings:
        st.error("No embeddings found or failed to parse embeddings.")
        return [], []
    embeddings = np.vstack(embeddings).astype('float32')
    return image_paths, embeddings

# Initialize FAISS index
@st.cache_resource
def create_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Streamlit UI
def main():
    st.title("Image Vector Database Tester")

    # Sidebar for database path
    st.sidebar.header("Database Configuration")
    db_path = st.sidebar.text_input("Path to `vector_database.db`", value="vector_database.db")

    if not db_path:
        st.error("Please provide the path to your `vector_database.db`")
        return

    # Establish database connection
    conn = get_db_connection(db_path)
    if not conn:
        return

    # List tables in the database
    tables = list_tables(conn)
    if not tables:
        st.error("No tables found in the database.")
        return

    # Sidebar for selecting table
    table_name = st.sidebar.selectbox("Select Table", options=tables)

    st.write(f"Using table: **{table_name}**")

    # Load embeddings and image paths
    image_paths, embeddings = load_embeddings(conn, table_name)

    if not image_paths or embeddings.size == 0:
        st.error("No data found in the selected table.")
        return

    # Initialize FAISS index
    index = create_faiss_index(embeddings)
    if index is None:
        st.error("Failed to create FAISS index. Check your embeddings.")
        return

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
        k = st.slider("Number of similar images to retrieve", min_value=1, max_value=10, value=5)
        distances, indices = index.search(image_embedding, k)

        st.subheader("Top Similar Images:")
        cols = st.columns(k)
        for i in range(k):
            idx = indices[0][i]
            sim_image_path = image_paths[idx]
            try:
                sim_image = Image.open(sim_image_path)
                cols[i].image(sim_image, caption=f"Similarity: {distances[0][i]:.2f}", use_column_width=True)
            except Exception as e:
                cols[i].write(f"Error loading image:\n{sim_image_path}\n{e}")

if __name__ == "__main__":
    main()
