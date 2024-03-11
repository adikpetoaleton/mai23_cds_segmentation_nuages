import streamlit as st
import random
import os

from utils import *
from utils_unet import load_model_stripe, load_model_unet, predict_and_display

st.set_page_config(page_title="Cloud-Net", page_icon=":five:")


@st.cache_resource()
def load_model_unet_cached():
    return load_model_unet()

@st.cache_resource()
def load_model_stripe_cached():
    return load_model_stripe()

st.sidebar.title('CloudIAtlas Project')
# Chargez l'image
image = 'utils\cloudiatlas.png'

# Ajoutez l'image à la barre latérale
st.sidebar.image(image, caption='Référence littéraire et cinématographique', use_column_width=True)
st.sidebar.divider()
st.sidebar.text("Avec la participation de :")
st.sidebar.markdown("***Alêton ADIKPETO***")
st.sidebar.markdown("***Soudjad CASSAM-CHENAI***")
st.sidebar.markdown("***Arnaud KREWERAS***")
st.sidebar.divider()
st.sidebar.text("Et comme guest stars :")
st.sidebar.markdown("***Aziz***")
st.sidebar.markdown("***Alban*** ")

model_unet = load_model_unet_cached()
model_stripe = load_model_stripe_cached()

st.title(":five: Modélisation avec U-Net")
display_info("Espace de test de l'architecture U-Net pour la segmentation sémantique des régions nuageuses")

image_folder = os.path.join(os.path.dirname(os.getcwd()), 'mai23_cds_segmentation_nuages\images')
image_list = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

if not image_list:
    st.write("Le dossier 'images' ne contient pas d'images .jpg. ET c'est pas bon ça !!!")
else:

    slide_col, text_col, random_col = st.columns([10,3,5])

    # Créer un slider pour choisir l'image
    slider = slide_col.slider('Choisissez une image', 1, len(image_list), 1)
    # selected_image = image_list[slider - 1]
    st.session_state.selected_image = image_list[slider - 1]
    text_col.text('ou')
    # predict_and_display(st.session_state.selected_image, model_unet, model_stripe)

    # Ajouter un bouton pour choisir une image au hasard
    if random_col.button('Choisir au hasard'):
        st.session_state.selected_image = random.choice(image_list)
        # predict_and_display(st.session_state.selected_image, model_unet, model_stripe)
        # selected_image = random.choice(image_list)

    # Afficher l'image sélectionnée et son nom
    st.image(os.path.join(image_folder, st.session_state.selected_image))
    predict_and_display(st.session_state.selected_image, model_unet, model_stripe)
    # st.write(f"Image sélectionnée : {selected_image}")

    # if st.button('Appliquer le modèle'):
    #     predict_and_display(st.session_state.selected_image, model_unet, model_stripe)
