import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load the saved model
model_path = r'C:\Users\AISU\OneDrive\Documents\Animal Emotion\time.keras'
model = load_model(model_path)
print("Model loaded successfully.")

# Emotion labels
emotion_classes = ['Angry', 'Happy', 'Sad']

# Function to check valid image types using Pillow
def is_image(file_path):
    try:
        Image.open(file_path)
        return True
    except IOError:
        return False

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path and is_image(file_path):
        img = Image.open(file_path)
        img = img.resize((224, 224), Image.LANCZOS)  # Resize using LANCZOS filter
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        predict_image(file_path)
    else:
        messagebox.showerror("Error", "Invalid file selected or no file chosen.")

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.LANCZOS)  # Resize using LANCZOS filter
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    print("Input image shape:", img_array.shape)  # Debug print
    
    with tf.device('/CPU:0'):  # Use CPU for prediction to avoid GPU memory issues in GUI apps
        outputs = model.predict(img_array)
    
    print("Model prediction outputs:", outputs)  # Debug print
    
    emotion_idx = np.argmax(outputs, axis=1)[0]
    emotion = emotion_classes[emotion_idx]
    
    result_label.config(text=f"Predicted Emotion: {emotion}")

# Tkinter setup
root = tk.Tk()
root.title("Animal Emotion Predictor")

upload_button = tk.Button(root, text="Upload Image", command=load_image)
upload_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Predicted Emotion: ")
result_label.pack(pady=10)

root.mainloop()











