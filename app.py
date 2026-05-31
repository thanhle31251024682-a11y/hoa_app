import gradio as gr
import numpy as np
import json
from PIL import Image
import os

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Try to load model
try:
    import tensorflow as tf
    model = tf.keras.models.load_model("flower_model.h5")
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    MODEL_LOADED = False

def predict_flower(image):
    if image is None:
        return {}
    
    if not MODEL_LOADED:
        return {"Error: Model not loaded": 1.0}
    
    # Preprocess image
    img = Image.fromarray(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    
    # Map to labels
    if isinstance(class_labels, dict):
        labels = [class_labels[str(i)] for i in range(len(predictions))]
    else:
        labels = class_labels
    
    result = {label: float(pred) for label, pred in zip(labels, predictions)}
    return result

# Custom CSS for beautiful UI
css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bloom-rose: #E8A598;
    --bloom-sage: #8FAF8C;
    --bloom-cream: #FAF6F0;
    --bloom-dark: #2C2420;
    --bloom-gold: #C9A96E;
    --bloom-mist: #D4E4D0;
}

body, .gradio-container {
    background-color: var(--bloom-cream) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    color: var(--bloom-dark) !important;
    font-size: 2.8rem !important;
    text-align: center !important;
    letter-spacing: -0.5px !important;
}

.subtitle {
    text-align: center;
    color: #7A6E68;
    font-size: 1.05rem;
    margin-top: -8px;
    margin-bottom: 24px;
    font-weight: 300;
}

.upload-area {
    border: 2px dashed var(--bloom-rose) !important;
    border-radius: 16px !important;
    background: white !important;
}

.btn-primary {
    background: var(--bloom-dark) !important;
    color: var(--bloom-cream) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    background: var(--bloom-rose) !important;
    transform: translateY(-1px) !important;
}

label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--bloom-dark) !important;
}

footer { display: none !important; }
"""

# Example images (if available)
examples = []
for ext in ["jpg", "jpeg", "png"]:
    for f in os.listdir(".") if os.path.isdir(".") else []:
        if f.endswith(ext):
            examples.append(f)

with gr.Blocks(css=css, title="🌸 Flower Identifier") as demo:
    gr.HTML("""
        <div style="padding: 40px 0 8px 0;">
            <h1>🌸 Flower Identifier</h1>
            <p class="subtitle">Upload a photo of a flower and discover what it is</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Flower Photo",
                type="numpy",
                elem_classes=["upload-area"],
                height=320
            )
            submit_btn = gr.Button(
                "✦ Identify Flower",
                elem_classes=["btn-primary"],
                variant="primary"
            )
        
        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Prediction Results",
                num_top_classes=5,
            )
    
    submit_btn.click(
        fn=predict_flower,
        inputs=image_input,
        outputs=label_output
    )
    
    image_input.change(
        fn=predict_flower,
        inputs=image_input,
        outputs=label_output
    )
    
    gr.HTML("""
        <div style="text-align:center; padding: 24px 0 8px; color: #9E9189; font-size: 0.85rem; font-family: 'DM Sans', sans-serif;">
            Powered by TensorFlow · Built with Gradio
        </div>
    """)

if __name__ == "__main__":
    demo.launch()
