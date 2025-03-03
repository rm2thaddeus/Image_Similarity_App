import os
import streamlit as st
import faiss
import pickle
import clip
import torch
from PIL import Image
import numpy as np

@st.cache_resource
def load_clip_model():
    """
    Loads the CLIP model and its preprocessing function.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def get_image_files(folder_path):
    """
    Scans the specified folder for image files with common extensions.
    """
    import glob
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_files

def extract_embeddings(model, preprocess, device, image_paths):
    """
    Extracts CLIP embeddings for a list of images.
    """
    embeddings = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                # Normalize the embedding
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().numpy())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    if embeddings:
        return np.vstack(embeddings)
    return np.array([])

def build_faiss_index(folder_path, model, preprocess, device):
    """
    Build a FAISS index from images in the specified folder and return the index + metadata.
    """
    image_files = get_image_files(folder_path)
    if not image_files:
        return None, []

    # Extract embeddings
    embeddings = extract_embeddings(model, preprocess, device, image_files)
    if embeddings.size == 0:
        return None, []

    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    # Prepare metadata (list of dicts with image paths)
    metadata = [{'image_path': path} for path in image_files]
    return index, metadata

def load_or_create_index(folder_path, model, preprocess, device):
    """
    Loads a FAISS index and metadata if they exist. Otherwise creates them.
    Returns (index, metadata).
    """
    index_file = os.path.join(folder_path, "image_index.faiss")
    metadata_file = os.path.join(folder_path, "metadata.pkl")

    # Check if index and metadata exist
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        # Load index
        index = faiss.read_index(index_file)
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        # Build index
        index, metadata = build_faiss_index(folder_path, model, preprocess, device)
        if index is None or len(metadata) == 0:
            st.warning("No images found or no embeddings could be created.")
            return None, None
        # Save them to disk
        faiss.write_index(index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

    return index, metadata

def encode_text_query(model, device, text_query):
    """
    Encodes a text query using CLIP.
    """
    # Tokenize text
    text_tokens = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.cpu().numpy()

def encode_uploaded_image(model, preprocess, device, uploaded_file):
    """
    Encodes an uploaded image file using CLIP.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    return image, image_embedding.cpu().numpy()

def find_best_match(index, query_embedding, metadata, top_k=1):
    """
    Given an FAISS index, a query embedding (text or image), 
    returns the top_k matching images from metadata.
    """
    if not index:
        return []
    distances, indices = index.search(query_embedding, top_k)
    # indices is shape [N, top_k], distances same shape
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((metadata[idx]['image_path'], dist))
    return results

def main():
    st.title("Image Finder App")
    st.write("Provide the path to your image folder, then search by text or by uploading an image.")

    # Prompt user for folder path
    folder_path = st.text_input("Folder path containing images:", value="")

    model, preprocess, device = load_clip_model()

    if folder_path:
        # Attempt to load or build the index
        if "faiss_index" not in st.session_state or st.session_state.get("folder_path") != folder_path:
            st.session_state["faiss_index"], st.session_state["metadata"] = load_or_create_index(
                folder_path, model, preprocess, device
            )
            st.session_state["folder_path"] = folder_path

        index = st.session_state.get("faiss_index", None)
        metadata = st.session_state.get("metadata", [])

        if index is not None and metadata:
            st.subheader("Search by text")
            text_query = st.text_input("Type your text query here and press Enter:")
            if text_query:
                text_emb = encode_text_query(model, device, text_query)
                results = find_best_match(index, text_emb, metadata, top_k=1)
                if results:
                    best_match_path, distance = results[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Your text query**")
                        # We can just show the query or a placeholder image
                        st.write(f'"{text_query}"')
                    with col2:
                        st.write("**Closest match**")
                        st.image(best_match_path, use_column_width=True)
                        st.write(f"Distance: {distance:.4f}")

            st.subheader("Or search by uploading an image")
            uploaded_file = st.file_uploader("Upload an image to find the most similar image in the folder", type=["png","jpg","jpeg","bmp","gif"])
            if uploaded_file is not None:
                query_image, query_embedding = encode_uploaded_image(model, preprocess, device, uploaded_file)
                results = find_best_match(index, query_embedding, metadata, top_k=1)
                if results:
                    best_match_path, distance = results[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Uploaded Image**")
                        st.image(query_image, use_column_width=True)
                    with col2:
                        st.write("**Closest match from database**")
                        st.image(best_match_path, use_column_width=True)
                        st.write(f"Distance: {distance:.4f}")
        else:
            st.warning("No FAISS index or metadata available. Please check your folder path and try again.")

if __name__ == "__main__":
    main()
