import os
import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier_rf.joblib")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
classes = bundle["classes"]
img_size = bundle["img_size"]

# Color palette for classes
CLASS_COLORS = {
    "cardboard": "#8D6E63",
    "glass": "#26A69A",
    "metal": "#78909C",
    "organic": "#66BB6A",
    "paper": "#AB47BC",
    "plastic": "#42A5F5",
    "trash": "#EF5350"
}

class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classification App")
        self.root.geometry("520x680")
        self.root.resizable(False, False)
        self.root.configure(bg="#F4F6F8")

        # Header Title
        title_frame = tk.Frame(self.root, bg="#1E88E5", pady=12)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(
            title_frame, 
            text="Waste Classification AI", 
            font=("Segoe UI", 16, "bold"), 
            fg="white", 
            bg="#1E88E5"
        )
        title_label.pack()

        subtitle = tk.Label(
            title_frame, 
            text=f"Supports: {', '.join(classes)}", 
            font=("Segoe UI", 9), 
            fg="#E3F2FD", 
            bg="#1E88E5"
        )
        subtitle.pack()

        # Image Display Area
        self.img_frame = tk.Frame(self.root, bg="#FFFFFF", width=360, height=360, relief=tk.RIDGE, bd=2)
        self.img_frame.pack(pady=20)
        self.img_frame.pack_propagate(False)

        self.img_label = tk.Label(self.img_frame, text="No image selected\nClick 'Choose Image' below", bg="#FFFFFF", fg="#757575", font=("Segoe UI", 11))
        self.img_label.pack(expand=True)

        # Result Display Area
        self.result_frame = tk.Frame(self.root, bg="#F4F6F8")
        self.result_frame.pack(pady=5, fill=tk.X, padx=30)

        self.res_badge = tk.Label(
            self.result_frame, 
            text="Prediction: -", 
            font=("Segoe UI", 14, "bold"), 
            bg="#ECEFF1", 
            fg="#37474F", 
            padx=16, 
            pady=8,
            relief=tk.FLAT
        )
        self.res_badge.pack(fill=tk.X)

        self.conf_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 10),
            bg="#F4F6F8",
            fg="#546E7A",
            pady=4
        )
        self.conf_label.pack()

        # Action Buttons
        btn_frame = tk.Frame(self.root, bg="#F4F6F8")
        btn_frame.pack(pady=15)

        self.btn_select = tk.Button(
            btn_frame, 
            text="📁 Select Waste Image", 
            command=self.select_image,
            font=("Segoe UI", 11, "bold"),
            bg="#1E88E5",
            fg="white",
            activebackground="#1565C0",
            activeforeground="white",
            padx=16,
            pady=8,
            cursor="hand2",
            relief=tk.FLAT
        )
        self.btn_select.pack(side=tk.LEFT, padx=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Waste Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        try:
            # Read image for display
            display_img = Image.open(file_path).convert("RGB")
            display_img.thumbnail((340, 340))
            photo = ImageTk.PhotoImage(display_img)
            self.img_label.configure(image=photo, text="")
            self.img_label.image = photo

            # Predict
            cv_img = cv2.imread(file_path)
            if cv_img is None:
                messagebox.showerror("Error", "Could not read selected image file.")
                return

            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv_img = cv2.resize(cv_img, (img_size, img_size))
            features = (cv_img.flatten().astype(np.float32) / 255.0).reshape(1, -1)

            prediction_idx = model.predict(features)[0]
            predicted_class = classes[prediction_idx]

            # Probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = probs[prediction_idx] * 100
                top3_indices = np.argsort(probs)[::-1][:3]
                top3_str = " | ".join([f"{classes[i]}: {probs[i]*100:.1f}%" for i in top3_indices])
            else:
                confidence = 100.0
                top3_str = ""

            # Update UI badge
            bg_color = CLASS_COLORS.get(predicted_class, "#455A64")
            self.res_badge.configure(
                text=f"Waste Type: {predicted_class.upper()} ({confidence:.1f}%)",
                bg=bg_color,
                fg="white"
            )
            self.conf_label.configure(text=f"Top Predictions: {top3_str}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify image: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassifierApp(root)
    root.mainloop()
