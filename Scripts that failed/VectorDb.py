import os
import sqlite3
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_jpeg_paths(folder_path):
    """
    Scans the folder for JPEG images and returns a list of their paths.
    """
    jpeg_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpeg_paths.append(os.path.join(root, file))
    return jpeg_paths

def generate_embedding(image_path):
    """
    Generates an embedding for the given image using CLIP.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding[0].numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def store_embeddings(db_path, image_data):
    """
    Stores image paths and embeddings in an SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        embedding BLOB
    )
    ''')
    
    # Insert data into the database
    for data in image_data:
        if data['embedding'] is not None:  # Only store if embedding is successful
            cursor.execute('''
            INSERT INTO Embeddings (image_path, embedding) VALUES (?, ?)
            ''', (data['path'], data['embedding'].tobytes()))  # Store embeddings as bytes
    
    conn.commit()
    conn.close()

def main(folder_path, db_path):
    # Step 1: Get JPEG paths
    image_paths = get_jpeg_paths(folder_path)
    
    # Step 2: Generate embeddings for each image
    image_data = [{'path': path, 'embedding': generate_embedding(path)} for path in image_paths]
    
    # Step 3: Store embeddings in the database
    store_embeddings(db_path, image_data)
    print(f"Embeddings stored in {db_path}")

# Example usage:
folder_path = r"C:\Users\aitor\OneDrive\Escritorio\Mael Post"  # Ensure paths are correct
db_path = r"C:\Users\aitor\OneDrive\Escritorio\Mael Post\vector_database.db"  # Ensure paths are correct
main(folder_path, db_path)
