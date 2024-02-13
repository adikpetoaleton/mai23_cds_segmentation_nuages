import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import os
from utils import *
import io

st.set_page_config(
    layout='centered'
)

@st.cache_data
def load_data():
    data = pd.read_csv('clouds_cleaned_sample.csv')
    return data

@st.cache_data
def load_data_2():
    data = pd.read_csv('clouds_enriched_sample.csv')
    return data

# Initialisation des variables de session
if 'dataframe_2' not in st.session_state:
    st.session_state['dataframe_2'] = None

if 'isextended' not in st.session_state:
    st.session_state['isextended'] = False

if 'issampled' not in st.session_state:
    st.session_state['issampled'] = False

st.title("Préparation des données")
display_info("Cette phase consiste à préparer le jeu de données en conjonction avec les images\
             réduites de sorte qu'il soit exploitable pour l'étape de modélisation.")

# Chargement du jeu de données initiale
if 'dataframe_2' in st.session_state:
    if st.session_state['dataframe_2'] is None:
        st.session_state['dataframe_2'] = load_data()
    df = st.session_state['dataframe_2']

###################################
# Aperçu du jeu de données brut 2 #
###################################

st.markdown("### 1. Aperçu du jeu de données brut")
st.dataframe(df.head())

###########################
# Préparation des données #
###########################

st.markdown("### 2. Préparation des données")
display_info("Cette étape consiste à labélliser le jeu de données à l'aide des coordonnées des Bounding Box.")

# Chargement des données pré-traitées (c'est plus rapide)
if st.checkbox("Données pré-traitées (rapide)"):

    if st.button('Charger') or st.session_state['isextended']:

        if 'dataframe_2' in st.session_state:
            st.session_state['dataframe_2'] = load_data_2()
            df = st.session_state['dataframe_2']

        if 'isextended' in st.session_state:
            st.session_state['isextended'] = True

        st.dataframe(df.head())

# On relance la préparation des données   
else:
   if st.button('Démarrer'):
    
        boxes = []
        for imageid, pixels_count in zip(df['ImageId'], df['EncodedPixelsCount']):
            imageid, bbox, resized_mask, resized_img, mask_pixels, box_pixels = get_single_image_bounding_box(data=df, imageid=imageid, image_path='images/', img_width=2100, img_height=1400, resize=(525, 350), pixels_count=pixels_count)
            boxes.append(
                {
                    'ImageId': imageid, 
                    'X': bbox['X'], 
                    'Y': bbox['Y'], 
                    'W': bbox['W'], 
                    'H': bbox['H'], 
                    'ResizedMaskPixelsCount': mask_pixels, 
                    'BoundingBoxPixelsCount': box_pixels, 
                    'BoxMaskGap': (box_pixels - mask_pixels) / mask_pixels
                }
            )
        bounding_boxes = pd.DataFrame(boxes)
        df = df.merge(right=bounding_boxes, on='ImageId', how='inner')
        df = df.reindex(columns=['ImageId', 'FileId', 'EncodedPixels', 'EncodedPixelsCount', 'X', 'Y', 'W', 'H', 'ResizedMaskPixelsCount', 'BoundingBoxPixelsCount', 'BoxMaskGap', 'Label', 'Class'])
        df = df.sort_values(by='BoxMaskGap', ascending=False)
        
        file_path = 'clouds_2.csv'
        if not(os.path.exists(file_path)):
            df.to_csv('clouds_2.csv', index=False)
        
        if 'dataframe_2' in st.session_state:
            st.session_state['dataframe_2'] = df

        if 'isextended' in st.session_state:
            st.session_state['isextended'] = True

        st.dataframe(df.head())

if st.session_state['isextended']:

    # Informations sur le dataset
    if st.checkbox("Afficher les informations", key=1):
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        info_output = info_buffer.getvalue()
        st.text(info_output)

    # Echantillonage des images avec classes non-disjointes
    st.markdown("##### 2.1 Echantillonage des images avec classes non-disjointes")
    display_info("Cette étape consiste à sélectionner les images qui ne possèdent qu'une seule classe\
                 et pour lesquelles le segment n'est pas discontinu.")

    GAP_LIMIT_LOW = 0
    GAP_LIMIT_HIGH = 0.09 # (< 0.09 -> 9%)

    df = df[(df['BoxMaskGap'] >= GAP_LIMIT_LOW) & (df['BoxMaskGap'] < GAP_LIMIT_HIGH)]

    if 'dataframe_2' in st.session_state:
        st.session_state['dataframe_2'] = df

    if st.button('Afficher', key=3):
        st.dataframe(df.head())
        if 'issampled' in st.session_state:
            st.session_state['issampled'] = True

    if st.session_state['issampled']:

        if st.checkbox("Afficher les informations", key=2) :
            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            info_output = info_buffer.getvalue()
            st.text(info_output)

###########################
# Exploration des données #
###########################

if st.session_state['isextended']:
    
    st.markdown("### 3. Exploration des données")
    display_info("Les graphique ci-après nous donne une idée de la répartition dans l'espace\
                 des Bouding Boxes et de leur taille moyenne.")

    tab_a, tab_b = st.tabs(['Répartition dans l\'espace', 'Taille moyenne'])
    
    with tab_a:
        fig1 = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.hist(df.X, range=(0, 499), bins=20, rwidth=0.8)
        plt.xlabel('X')
        plt.ylabel('Count')
        plt.subplot(122)
        plt.hist(df.Y, range=(0, 349), bins=20, rwidth=0.8)
        plt.xlabel('Y')
        plt.ylabel('Count')
        st.pyplot(fig1)

    with tab_b:
        fig2 = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.hist(df.W, range=(0, 499), bins=20, rwidth=0.8)
        plt.xlabel('W')
        plt.ylabel('Count')
        plt.subplot(122)
        plt.hist(df.H, range=(0, 349), bins=20, rwidth=0.8)
        plt.xlabel('H')
        plt.ylabel('Count')
        st.pyplot(fig2)

############################
# Visualisation des images #
############################

if st.session_state['isextended']:
    
    st.markdown("### 4. Visualisation des images")
    display_info("Cette étape consiste à afficher aléatoirement 3 images d'origine (2100 x 1400)\
                 avec leur équivalance en taille réduite (515 x 350) en mettant en évidance les\
                 Bounding boxes respectives de ces dernières. Le but est de vérifier la capacité du système à marquer\
                 correctement les segments prédits.")

    if st.button("Afficher", key=4):
        with st.spinner("Veuillez patienter"):

            img_width = 2100
            img_height = 1400
            cmap = 'viridis'
            alpha = 0.2
            image_path = 'images/'
            ImageIds = random.sample(df['ImageId'].unique().tolist(), 3) # Use this line (but comment the previous) to let the system select images at random

            fig, axes = plt.subplots(2, 3, figsize=(17, 11),  layout='constrained')
            for axe, img_id, i in zip(axes.flat, ImageIds, range(3)):
                displayMask(img_id, axes.flat[i], df, img_width, img_height, cmap, alpha, image_path)

                x = df[df['ImageId'] == img_id].X
                y = df[df['ImageId'] == img_id].Y
                w = df[df['ImageId'] == img_id].W
                h = df[df['ImageId'] == img_id].H
                displayBoundingBox(img_id, axes.flat[i + 3], x, y, w, h)
            st.pyplot(fig)
