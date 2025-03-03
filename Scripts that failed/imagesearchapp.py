import os
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
import sqlite3
import faiss
from transformers import CLIPProcessor, CLIPModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_embeddings_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT embedding, image_path FROM embeddings")
    data = cursor.fetchall()
    embeddings = []
    image_paths = []
    for row in data:
        # Assuming the embedding is stored as a binary blob
        embedding = np.frombuffer(row[0], dtype=np.float32)
        embeddings.append(embedding)
        image_paths.append(row[1])
    embeddings = np.vstack(embeddings)
    conn.close()
    return embeddings, image_paths
def build_search_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
class ImageSearchApp:
    def __init__(self, master, index, image_paths):
        self.master = master
        self.index = index
        self.image_paths = image_paths
        self.setup_gui()

    def setup_gui(self):
        self.master.title("Image Search Chatbox")
        self.chat_label = tk.Label(self.master, text="Enter your query:")
        self.chat_label.pack()
        self.query_entry = tk.Entry(self.master, width=50)
        self.query_entry.pack()
        self.search_button = tk.Button(self.master, text="Search", command=self.perform_search)
        self.search_button.pack()
        self.message_label = tk.Label(self.master, text="")
        self.message_label.pack()
        self.image_window = tk.Toplevel(self.master)
        self.image_window.title("Search Results")

    def perform_search(self):
        query = self.query_entry.get()
        retrieved_images = self.search_images(query)
        self.display_results(retrieved_images)
        description = self.generate_description(retrieved_images, query)
        self.message_label.config(text=description)

    def search_images(self, query):
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**inputs)
        text_embedding = text_embedding.cpu().numpy()
        distances, indices = self.index.search(text_embedding, k=5)
        retrieved_paths = [self.image_paths[idx] for idx in indices[0]]
        return retrieved_paths

    def display_results(self, image_paths):
        for widget in self.image_window.winfo_children():
            widget.destroy()
        for img_path in image_paths:
            img = Image.open(img_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            label = tk.Label(self.image_window, image=img_tk)
            label.image = img_tk  # Keep a reference
            label.pack()

    def generate_description(self, image_paths, query):
        return f"The system found {len(image_paths)} images related to '{query}'."
def main():
    # Assuming the script and vector_database.db are in the same folder
    db_path = os.path.join(os.getcwd(), "vector_database.db")
    embeddings, image_paths = load_embeddings_from_db(db_path)
    index = build_search_index(embeddings)
    root = tk.Tk()
    app = ImageSearchApp(root, index, image_paths)
    root.mainloop()

if __name__ == "__main__":
    main()
