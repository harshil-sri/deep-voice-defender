import os
import sys

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import sys
import os

# Ensure src directory is in path so we can import preprocessing modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_audio, audio_to_mel

# ==========================================
# STREAMLIT CONFIG & THEME SETUP
# ==========================================
st.set_page_config(
    page_title="Deep-Voice Defender",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to force a "Corporate Cyber-Forensics" dark aesthetic
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00e6e6;
        text-shadow: 0px 0px 8px rgba(0, 230, 230, 0.4);
    }
    
    /* Upload Box */
    .stFileUploader > div > div {
        background-color: #151b2b;
        border: 1px solid #00e6e6;
        border-radius: 5px;
    }
    
    /* Metric / Verdict Box */
    .metric-container {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        margin-top: 20px;
    }
    
    /* Warning (Spoof) vs Success (Bonafide) Styles */
    .verdict-spoof {
        color: #ff4d4d;
        text-shadow: 0px 0px 10px rgba(255, 77, 77, 0.8);
        font-size: 32px;
        font-weight: 900;
        letter-spacing: 2px;
    }
    .verdict-bonafide {
        color: #00ffcc;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.8);
        font-size: 32px;
        font-weight: 900;
        letter-spacing: 2px;
    }
    .confidence-score {
        font-size: 20px;
        color: #9ca3af;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_deepvoice_model():
    """Loads the Keras model once and caches it to prevent reloading."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), "deep_voice_defender_v1.keras")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Model load error: {e}. Please ensure 'deep_voice_defender_v1.keras' exists.")
        return None

def plot_spectrogram(spectrogram):
    """Renders a visually striking Log-Mel Spectrogram."""
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    
    # Plot standard spectrogram with a 'magma' colormap for high-tech look
    img = librosa.display.specshow(
        spectrogram, 
        x_axis='time', 
        y_axis='mel', 
        sr=16000, 
        ax=ax, 
        cmap='magma'
    )
    
    # Stylize axes
    ax.tick_params(colors='#e0e6ed')
    ax.xaxis.label.set_color('#e0e6ed')
    ax.yaxis.label.set_color('#e0e6ed')
    ax.set_title('Log-Mel Spectrogram Analysis', color='#00e6e6')
    
    fig.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(colors='#e0e6ed')
    plt.tight_layout()
    return fig


# ==========================================
# MAIN APP LAYOUT
# ==========================================
def main():
    st.markdown("<h1>🛡️ DEEP-VOICE DEFENDER</h1>", unsafe_allow_html=True)
    st.markdown("### Forensic Audio Analysis Terminal")
    st.markdown("---")

    # Load Model
    model = load_deepvoice_model()
    if model is None:
        return

    # Sidebar / Information
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        st.success("🟢 Engine Online")
        st.success("🟢 Model Loaded")
        st.markdown("---")
        st.markdown("**Version:** 1.0.0-beta")
        st.markdown("**Author:** Deep-Voice Defender Team")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload Suspect Audio File", type=['wav', 'flac'])

    if uploaded_file is not None:
        # Columns for layout (Audio Player | Processing Stats)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎧 Suspect Audio")
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        # We need to save the file temporarily to use librosa load
        temp_file_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        
        # Create an Analyze button
        if st.button("🔍 INITIATE FORENSIC SCAN", use_container_width=True):
            with st.spinner("Analyzing audio frequencies..."):
                start_time = time.time()
                
                # --- Pipeline Execution ---
                # 1. Load & Pad Audio
                raw_audio = load_audio(temp_file_path)
                
                # 2. Extract Spectrogram
                spectrogram = audio_to_mel(raw_audio)
                
                # 3. Predict
                input_data = np.expand_dims(spectrogram, axis=0) # Add batch dim
                input_data = np.expand_dims(input_data, axis=-1) # Add channel dim
                
                prediction = model.predict(input_data, verbose=0)
                score = prediction[0][0]
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # --- Rendering the Results ---
                st.markdown("### 🔬 Visual Forensics")
                # Plot Spectrogram
                fig = plot_spectrogram(spectrogram)
                st.pyplot(fig)
                
                # Verdict Dashboard
                st.markdown("### 📑 Final Verdict Dashboard")
                
                verdict_class = "SPOOF (AI GENERATED)" if score < 0.5 else "BONAFIDE (HUMAN SPEECH)"
                confidence = (1 - score) * 100 if score < 0.5 else score * 100
                css_class = "verdict-spoof" if score < 0.5 else "verdict-bonafide"
                
                # Custom HTML block for Verdict
                html_str = f"""
                <div class="metric-container">
                    <p style="color:#9ca3af; font-size:18px; margin-bottom:0;">DETECTED SIGNATURE:</p>
                    <p class="{css_class}">{verdict_class}</p>
                    <p class="confidence-score">Confidence: {confidence:.2f}% | Raw Score: {score:.4f}</p>
                    <p style="color:#6b7280; font-size:12px; margin-top:10px;">Inference Latency: {latency_ms:.1f} ms</p>
                </div>
                """
                st.markdown(html_str, unsafe_allow_html=True)
                
                # Cleanup temp file
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
