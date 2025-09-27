import streamlit as st
import numpy as np
from green_gan_module import GreenGANTrainer  # Import your GAN trainer class

# Cache model loading so it’s done once
@st.cache(allow_output_mutation=True)
def load_model():
    gan = GreenGANTrainer(feature_dim=78, noise_dim=100)
    gan.load_models()  # Load pre-trained generator and discriminator
    return gan

gan = load_model()

st.title("Green-GAN: Synthetic Cyberattack Generator")

num_samples = st.slider("Number of Synthetic Attacks to Generate", min_value=100, max_value=5000, step=100, value=500)

if st.button("Generate Attacks"):
    with st.spinner("Generating synthetic attack vectors..."):
        synthetic_attacks = gan.generate_synthetic_attacks(num_samples)
        st.success(f"Generated {num_samples} synthetic attack vectors!")
        st.dataframe(np.array(synthetic_attacks)[:10])  # Show first 10 vectors in table


