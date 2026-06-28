import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Configuration ───
MODEL_PATH = os.path.join(os.path.dirname(__file__), "CustomCNN_best.keras")
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ─── Load Model ───
print("Loading model...")
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded. Input shape: {model.input_shape}")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}. Please place your trained model file here.")
    model = None

# ─── Grad-CAM Implementation ───
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generate Grad-CAM heatmap for the given image."""
    if last_conv_layer_name is None:
        # Find the last Conv2D layer
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        return None
    
    # Create a model that maps the input to the last conv layer and output
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradients of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature map by importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image."""
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to RGB heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)
    
    # Convert grayscale to RGB for overlay
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)
    else:
        img_rgb = img
    
    img_rgb = np.uint8(img_rgb * 255) if img_rgb.max() <= 1 else np.uint8(img_rgb)
    
    # Superimpose
    superimposed = heatmap * alpha + img_rgb * (1 - alpha)
    superimposed = np.uint8(superimposed)
    
    return superimposed


# ─── Preprocessing ───
def preprocess_image(image):
    """Convert uploaded image to 48x48 grayscale normalized array."""
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize to 48x48
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add channel and batch dimensions: (1, 48, 48, 1)
    img_array = np.expand_dims(img, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img


# ─── Prediction ───
def predict_emotion(image):
    """Main inference function for Gradio interface."""
    if model is None:
        return "Error: Model not loaded. Please check CustomCNN_best.keras exists.", None, None
    
    # Preprocess
    img_array, img_gray = preprocess_image(image)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_emotion = EMOTIONS[predicted_idx]
    confidence = float(predictions[predicted_idx])
    
    # Build label dict for gr.Label and confidence markdown
    label_output = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, predictions)}

    confidence_lines = []
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        marker = " ← PREDICTED" if i == predicted_idx else ""
        confidence_lines.append(f"{emotion:>8}: {bar} {prob:.2%}{marker}")

    confidence_text = f"**Predicted: {predicted_emotion}** (confidence: {confidence:.2%})\n\n"
    confidence_text += "All class probabilities:\n"
    confidence_text += "\n".join(confidence_lines)
    
    # Generate Grad-CAM
    try:
        heatmap = make_gradcam_heatmap(img_array, model, pred_index=predicted_idx)
        if heatmap is not None:
            gradcam_img = overlay_gradcam(img_gray, heatmap)
        else:
            gradcam_img = None
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gradcam_img = None
    
    # Convert 48x48 grayscale to displayable image
    display_img = np.uint8(img_gray * 255)
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
    
    return label_output, confidence_text, gradcam_img, display_img_rgb


# ─── Gradio Interface ───
def create_demo():
    with gr.Blocks(theme=gr.themes.Soft(), title="FER2013 Demo") as demo:
        gr.Markdown(
            """
            # 😊 Facial Expression Recognition Demo
            
            Upload a face image and the model will classify the emotion into one of **7 categories**:
            **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.
            
            The model is a **Custom CNN** trained on the FER2013 dataset (64.47% accuracy).
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 Upload Face Image",
                    type="numpy",
                    height=300
                )
                submit_btn = gr.Button("🔍 Predict Emotion", variant="primary")
                
                gr.Examples(
                    examples=[],
                    inputs=input_image,
                    label="Example images (upload your own above)"
                )
            
            with gr.Column(scale=1):
                prediction_label = gr.Label(
                    label="🎯 Predicted Emotion",
                    value="Upload an image to see prediction"
                )
                processed_image = gr.Image(
                    label="🖼️ Preprocessed (48×48 Grayscale)",
                    height=200
                )
        
        with gr.Row():
            confidence_text = gr.Markdown(label="📊 Confidence Scores")
        
        with gr.Row():
            gradcam_image = gr.Image(
                label="🔥 Grad-CAM Visualization",
                height=300
            )
        
        gr.Markdown(
            """
            ---
            **About this model:**
            - Architecture: Custom CNN (3 Conv blocks + Dense)
            - Input: 48×48 grayscale
            - Best class: Happy (F1 = 0.86) | Hardest: Fear (F1 = 0.44)
            - [GitHub](https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition) | [Kaggle](https://www.kaggle.com/code/lokeshchamakuri/facial-expression-recognition)
            
            **Tip:** For best results, upload a frontal face image with good lighting.
            """
        )
        
        # Event handler
        submit_btn.click(
            fn=predict_emotion,
            inputs=[input_image],
            outputs=[prediction_label, confidence_text, gradcam_image, processed_image]
        )
    
    return demo


# ─── Main Entry ───
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True
    )
