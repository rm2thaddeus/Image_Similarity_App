# prepare_database.py

import os
import glob
import pickle
from PIL import Image
import numpy as np
import faiss
import clip
import torch

def get_image_files(folder_path):
    """
    Scans the specified folder for image files with common extensions.
    
    Args:
        folder_path (str): Path to the folder to scan.
        
    Returns:
        list: List of image file paths.
    """
    # Define common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_files

def load_clip_model():
    """
    Loads the CLIP model and its preprocessing function.
    
    Returns:
        model: The CLIP model.
        preprocess: The preprocessing function for images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def extract_embeddings(model, preprocess, device, image_paths):
    """
    Extracts embeddings for a list of images using CLIP.
    
    Args:
        model: The CLIP model.
        preprocess: The preprocessing function for images.
        device (str): Device to run the model on ('cuda' or 'cpu').
        image_paths (list): List of image file paths.
        
    Returns:
        list: List of image embeddings.
    """
    embeddings = []
    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
                embeddings.append(embedding.cpu().numpy())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    if embeddings:
        return np.vstack(embeddings)
    else:
        return np.array([])

def create_faiss_index(embeddings, embedding_dim):
    """
    Creates a FAISS index from the given embeddings.
    
    Args:
        embeddings (np.ndarray): Array of image embeddings.
        embedding_dim (int): Dimension of the embeddings.
        
    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance
    index.add(embeddings)
    return index

def save_index(index, file_path):
    """
    Saves the FAISS index to a file.
    
    Args:
        index (faiss.IndexFlatL2): FAISS index.
        file_path (str): Path to save the index.
    """
    faiss.write_index(index, file_path)

def save_metadata(metadata, file_path):
    """
    Saves the metadata (e.g., image paths) to a file using pickle.
    
    Args:
        metadata (list): List of metadata dictionaries.
        file_path (str): Path to save the metadata.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(metadata, f)

def main():
    """
    Main function to prepare the image database.
    """
    # Specify the folder to scan (current directory by default)
    folder_path = os.getcwd()  # Change this if you want to specify a different folder
    print(f"Scanning folder: {folder_path}")
    
    # Get list of image files
    image_files = get_image_files(folder_path)
    print(f"Found {len(image_files)} image files.")
    
    if not image_files:
        print("No images found. Exiting.")
        return
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, preprocess, device, image_files)
    print(f"Extracted embeddings for {embeddings.shape[0]} images.")
    
    if embeddings.size == 0:
        print("No embeddings extracted. Exiting.")
        return
    
    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings, embedding_dim)
    
    # Save FAISS index
    index_file = "image_index.faiss"
    print(f"Saving FAISS index to {index_file}...")
    save_index(index, index_file)
    
    # Prepare and save metadata
    metadata = [{'image_path': path} for path in image_files]
    metadata_file = "metadata.pkl"
    print(f"Saving metadata to {metadata_file}...")
    save_metadata(metadata, metadata_file)
    import os
import glob
import pickle
import argparse
from PIL import Image
import numpy as np
import faiss
import clip
import torch

def get_image_files(folder_path):
    """ Scans the specified folder for image files. """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_files

def load_clip_model():
    """ Loads the CLIP model. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def extract_embeddings(model, preprocess, device, image_paths):
    """ Extracts CLIP embeddings for images. """
    embeddings = []
    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
                embeddings.append(embedding.cpu().numpy())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.vstack(embeddings) if embeddings else np.array([])

def create_faiss_index(embeddings):
    """ Creates a FAISS index from the embeddings. """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def save_index(index, file_path):
    """ Saves FAISS index to a file. """
    faiss.write_index(index, file_path)

def save_metadata(metadata, file_path):
    """ Saves metadata using pickle. """
    with open(file_path, 'wb') as f:
        pickle.dump(metadata, f)

def main():
    """ Main function to process images and create a FAISS index. """
    parser = argparse.ArgumentParser(description="Prepare the image database for CLIP-based search.")
    parser.add_argument("--folder", type=str, default=os.getcwd(), help="Path to the image folder")
    args = parser.parse_args()

    folder_path = args.folder
    print(f"Scanning folder: {folder_path}")

    # Get image files
    image_files = get_image_files(folder_path)
    print(f"Found {len(image_files)} image files.")
    
    if not image_files:
        print("No images found. Exiting.")
        return
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, preprocess, device, image_files)
    print(f"Extracted embeddings for {embeddings.shape[0]} images.")

    if embeddings.size == 0:
        print("No embeddings extracted. Exiting.")
        return

    # Create FAISS index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    # Save FAISS index and metadata
    index_file = os.path.join(folder_path, "image_index.faiss")
    metadata_file = os.path.join(folder_path, "metadata.pkl")
    print(f"Saving FAISS index to {index_file}...")
    save_index(index, index_file)

    metadata = [{'image_path': path} for path in image_files]
    print(f"Saving metadata to {metadata_file}...")
    save_metadata(metadata, metadata_file)

    print("Database preparation completed successfully.")

if __name__ == "__main__":
    main()

    print("Database preparation completed successfully.")

if __name__ == "__main__":
    main()
