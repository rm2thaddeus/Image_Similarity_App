import tkinter as tk
from tkinter import simpledialog, messagebox
import sqlite3
from PIL import Image, ImageTk
import os

# Connect to the database
conn = sqlite3.connect('c:/Users/aitor/OneDrive/Escritorio/Mael Post/vector_database.db')
cursor = conn.cursor()

# Function to query the database and display images
def query_database():
    # Prompt user for a description query
    query = simpledialog.askstring("Query Database", "Enter description to search for images:")

    if query:
        try:
            # Search for images matching the description
            cursor.execute("SELECT image_path FROM images WHERE description LIKE ?", ('%' + query + '%',))
            results = cursor.fetchall()

            if results:
                # Display images in a new window
                display_images([result[0] for result in results])
            else:
                messagebox.showinfo("No Results", "No images found matching the description.")
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"An error occurred: {e}")

# Function to display images in a new window
def display_images(image_paths):
    display_window = tk.Toplevel(root)
    display_window.title("Image Results")

    for path in image_paths:
        if os.path.exists(path):
            # Open the image
            img = Image.open(path)
            img.thumbnail((200, 200))  # Resize to fit in window
            img_tk = ImageTk.PhotoImage(img)

            # Create a label for each image
            label = tk.Label(display_window, image=img_tk)
            label.image = img_tk  # Keep a reference to avoid garbage collection
            label.pack(padx=5, pady=5)
        else:
            messagebox.showwarning("Missing File", f"Image file not found: {path}")

# Main application window
root = tk.Tk()
root.title("Image Query Application")

# Button to query the database
query_button = tk.Button(root, text="Query Database", command=query_database)
query_button.pack(padx=20, pady=20)

# Start the main application loop
root.mainloop()

# Close the database connection when the application exits
conn.close()
