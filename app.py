"""
User Interface
"""

import os
from io import BytesIO
import streamlit as st
import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import torch
from torchvision.datasets import ImageFolder
from lime.lime_image import LimeImageExplainer
from skimage.color import gray2rgb

from model.config import MasterConfig
from model.model import CustomCNN, Classifier, DEFAULT_TRANSFORM


def explain_prediction(image, target_class):
    example_image_rgb = gray2rgb(image)
    explainer = LimeImageExplainer()

    def predict_fn(rgb_image):
        grayscale_image = np.mean(rgb_image, axis=-1, keepdims=True)
        image = torch.tensor(grayscale_image).permute(0, 3, 1, 2)
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().numpy()

    explanation = explainer.explain_instance(
        example_image_rgb, predict_fn, top_labels=1, hide_color=0, num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        target_class, positive_only=True, num_features=1, hide_rest=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(temp)
    axes[1].imshow(mask, alpha=0.5, cmap="jet")
    axes[1].set_title(
        f"LIME Explanation for Class {target_class} - {MasterConfig.CLASS_NAMES[target_class]}"
    )
    axes[1].axis("off")
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    st.image(buf)


def preprocess_audio(path):
    audio, sr = librosa.load(path, sr=MasterConfig.SAMPLING_RATE)
    noise_profile = audio[: int(sr * 0.5)]
    reduced_noise = nr.reduce_noise(
        y=audio, sr=sr, y_noise=noise_profile, prop_decrease=1.0
    )
    non_silent_intervals = librosa.effects.split(reduced_noise, top_db=30)
    non_silent_audio = np.concatenate(
        [reduced_noise[start:end] for start, end in non_silent_intervals]
    )
    return non_silent_audio


def save_spectrogram(non_silent_audio, label, output_path="./tmp/"):
    stft = librosa.stft(non_silent_audio)
    stft_db = librosa.amplitude_to_db(np.abs(stft))
    plt.figure(figsize=(14, 5), dpi=400)
    librosa.display.specshow(
        stft_db,
        sr=MasterConfig.SAMPLING_RATE,
        x_axis=None,
        y_axis=None,
        vmin=MasterConfig.GLOBAL_MIN_DB,
        vmax=MasterConfig.GLOBAL_MAX_DB,
        cmap="gray",
    )
    plt.axis("off")
    uid = "example.png"
    if not os.path.exists(os.path.join(output_path, label)):
        os.makedirs(os.path.join(output_path, label), exist_ok=True)
    plt.savefig(
        os.path.join(output_path, label, uid),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def main():
    st.title("Speech recognition")
    choice = st.radio("Choose an option", ["Upload an audio file", "Record an audio"])
    if choice == "Upload an audio file":
        if os.path.exists("./tmp/unknown/example.png"):
            os.remove("./tmp/unknown/example.png")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        left, right = st.columns(2)
        with left:
            preview = st.checkbox("Preview audio")
        with right:
            explanation = st.checkbox("Generate explanation after prediction")
        if audio_file is not None:
            with NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            non_silent_audio = preprocess_audio(tmp_path)
            save_spectrogram(non_silent_audio, "unknown")

    if choice == "Record an audio":
        if os.path.exists("./tmp/unknown/example.png"):
            os.remove("./tmp/unknown/example.png")
        audio_file = st.audio_input("Record an audio")
        left, right = st.columns(2)
        with left:
            preview = st.checkbox("Preview audio")
        with right:
            explanation = st.checkbox("Generate explanation after prediction")
        if audio_file is not None:
            with NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            non_silent_audio = preprocess_audio(tmp_path)
            save_spectrogram(non_silent_audio, "unknown")

    if preview:
        if audio_file is None:
            st.error("Please upload/record an audio file first")
        else:
            st.audio(audio_file, format="audio/wav")
            img = (
                ImageFolder("./tmp/", transform=DEFAULT_TRANSFORM)[0][0]
                .squeeze(0)
                .numpy()
            )
            _, cent_co, _ = st.columns(3)
            with cent_co:
                st.image(img, caption="Spectrogram")

    predict_button = st.button("Predict")
    if predict_button:
        # predict the class of the audio file
        if audio_file is None:
            st.error("Please upload/record an audio file first")
        else:
            spectrogram = ImageFolder("./tmp/", transform=DEFAULT_TRANSFORM)
            image = spectrogram[0][0].unsqueeze(0)
            prediction_logits = model(image)
            prediction_probas = torch.softmax(prediction_logits, dim=1)
            prediction_class_idx = torch.argmax(prediction_probas, dim=1)
            prediction_proba = prediction_probas[0][prediction_class_idx].item()
            class_name = MasterConfig.CLASS_NAMES[prediction_class_idx]
            st.write(
                f"Prediction: {class_name} with probability {prediction_proba:.2f}"
            )
    if predict_button and explanation:
        with st.spinner("Generating explanation..."):
            explain_prediction(image.squeeze(0).squeeze(0), prediction_class_idx.item())


if __name__ == "__main__":
    model = torch.load(
        "./best_model/best-model-no-noise-no-silence.pkl",
        weights_only=False,
    )
    model.eval()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
