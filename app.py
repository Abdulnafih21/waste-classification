import os
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import webbrowser
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier_rf.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
classes = bundle["classes"]
img_size = bundle["img_size"]

CLASS_COLORS = {
    "cardboard": "#8D6E63",
    "glass": "#26A69A",
    "metal": "#78909C",
    "organic": "#4CAF50",
    "paper": "#AB47BC",
    "plastic": "#2196F3",
    "trash": "#F44336"
}

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Waste Classification AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background: #f0f4f8; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 16px; }
    .card { background: #ffffff; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); width: 100%; max-width: 600px; padding: 32px; text-align: center; }
    h1 { color: #1a202c; font-size: 26px; margin-bottom: 6px; }
    .sub { color: #718096; font-size: 14px; margin-bottom: 24px; }
    .dropzone { border: 2px dashed #cbd5e0; border-radius: 12px; padding: 36px 20px; cursor: pointer; transition: 0.2s; background: #fafbfc; position: relative; }
    .dropzone:hover, .dropzone.dragover { border-color: #3182ce; background: #ebf8ff; }
    .dropzone p { color: #4a5568; font-weight: 500; font-size: 15px; }
    .preview-container { margin-top: 20px; display: none; }
    .preview-img { max-width: 100%; max-height: 280px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); object-fit: contain; }
    .btn { display: inline-block; background: #3182ce; color: #fff; font-weight: 600; padding: 12px 28px; border-radius: 8px; border: none; cursor: pointer; font-size: 15px; margin-top: 18px; transition: 0.2s; }
    .btn:hover { background: #2b6cb0; }
    .btn:disabled { background: #a0aec0; cursor: not-allowed; }
    .result-box { margin-top: 24px; padding: 18px; border-radius: 12px; display: none; }
    .res-badge { font-size: 22px; font-weight: 700; color: white; display: inline-block; padding: 8px 24px; border-radius: 20px; margin-bottom: 12px; text-transform: uppercase; }
    .prob-list { list-style: none; margin-top: 14px; text-align: left; }
    .prob-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 13px; color: #4a5568; font-weight: 600; }
    .bar-bg { width: 60%; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; background: #3182ce; }
    .spinner { display: none; margin: 16px auto; width: 32px; height: 32px; border: 4px solid #e2e8f0; border-top: 4px solid #3182ce; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="card">
    <h1>Waste Classification AI</h1>
    <p class="sub">AI model trained across 7 balanced waste classes</p>

    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
      <p id="dropText">📁 Drag & drop an image here or click to browse</p>
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
    </div>

    <div class="preview-container" id="previewContainer">
      <img id="preview" class="preview-img" alt="Preview">
    </div>

    <button id="classifyBtn" class="btn" style="display:none" onclick="classifyImage()">Classify Waste</button>
    <div id="spinner" class="spinner"></div>

    <div id="resultBox" class="result-box">
      <div id="resBadge" class="res-badge"></div>
      <p id="resConfidence" style="color: #4a5568; font-weight: 600; font-size: 14px;"></p>
      <ul id="probList" class="prob-list"></ul>
    </div>
  </div>

  <script>
    let selectedFile = null;
    const dropzone = document.getElementById('dropzone');

    ['dragenter', 'dragover'].forEach(e => dropzone.addEventListener(e, (evt) => { evt.preventDefault(); dropzone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(e => dropzone.addEventListener(e, (evt) => { evt.preventDefault(); dropzone.classList.remove('dragover'); }));
    dropzone.addEventListener('drop', (evt) => {
      if (evt.dataTransfer.files && evt.dataTransfer.files[0]) {
        handleFile(evt.dataTransfer.files[0]);
      }
    });

    function handleFile(file) {
      if (!file) return;
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        document.getElementById('preview').src = e.target.result;
        document.getElementById('previewContainer').style.display = 'block';
        document.getElementById('classifyBtn').style.display = 'inline-block';
        document.getElementById('resultBox').style.display = 'none';
        document.getElementById('dropText').innerText = "Selected: " + file.name;
      };
      reader.readAsDataURL(file);
    }

    async function classifyImage() {
      if (!selectedFile) return;
      const formData = new FormData();
      formData.append('image', selectedFile);

      const spinner = document.getElementById('spinner');
      const classifyBtn = document.getElementById('classifyBtn');
      spinner.style.display = 'block';
      classifyBtn.disabled = true;

      try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        spinner.style.display = 'none';
        classifyBtn.disabled = false;

        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }

        const resultBox = document.getElementById('resultBox');
        const badge = document.getElementById('resBadge');
        const conf = document.getElementById('resConfidence');
        const probList = document.getElementById('probList');

        badge.innerText = data.predicted_class;
        badge.style.backgroundColor = data.color;
        conf.innerText = `Prediction Confidence: ${data.confidence.toFixed(1)}%`;

        probList.innerHTML = '';
        data.probabilities.forEach(p => {
          const row = document.createElement('li');
          row.className = 'prob-row';
          row.innerHTML = `
            <span>${p.class}</span>
            <div class="bar-bg">
              <div class="bar-fill" style="width: ${p.prob}%; background-color: ${p.color};"></div>
            </div>
            <span>${p.prob.toFixed(1)}%</span>
          `;
          probList.appendChild(row);
        });

        resultBox.style.display = 'block';
      } catch (err) {
        spinner.style.display = 'none';
        classifyBtn.disabled = false;
        alert('Request failed: ' + err.message);
      }
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        features = (img.flatten().astype(np.float32) / 255.0).reshape(1, -1)

        pred_idx = model.predict(features)[0]
        pred_class = classes[pred_idx]

        probs = model.predict_proba(features)[0]
        prob_list = []
        for i, c in enumerate(classes):
            prob_list.append({
                "class": c,
                "prob": float(probs[i] * 100),
                "color": CLASS_COLORS.get(c, "#3182ce")
            })
        prob_list.sort(key=lambda x: x["prob"], reverse=True)

        return jsonify({
            "predicted_class": pred_class,
            "confidence": float(probs[pred_idx] * 100),
            "color": CLASS_COLORS.get(pred_class, "#3182ce"),
            "probabilities": prob_list
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)

