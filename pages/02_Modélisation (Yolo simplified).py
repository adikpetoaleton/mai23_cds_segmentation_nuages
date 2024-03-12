import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, LeakyReLU, Dropout, Reshape, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

import os
import random

import matplotlib.gridspec as gridspec

from utility_for_modeling import *

import io

GRAPHS_PATH = 'images/graphs/'

st.set_page_config(page_title="Cloud Yolo", page_icon=":four:", layout='centered')

# st.sidebar.title('CloudIAtlas Project')
# # Chargez l'image
# image = 'utils\cloudiatlas.png'

# # Ajoutez l'image à la barre latérale
# st.sidebar.image(image, caption='Référence littéraire et cinématographique', use_column_width=True)
# st.sidebar.divider()
# st.sidebar.text("Avec la participation de :")
# st.sidebar.markdown("***Alêton ADIKPETO***")
# st.sidebar.markdown("***Soudjad CASSAM-CHENAI***")
# st.sidebar.markdown("***Arnaud KREWERAS***")
# st.sidebar.divider()
# st.sidebar.text("Et comme guest stars :")
# st.sidebar.markdown("***Aziz***")
# st.sidebar.markdown("***Alban*** ")

tf.keras.backend.clear_session()



#########################################
# Initialisation des variables globales #
#########################################

RESIZED_PATH = 'images/resized/'
IMAGES_PATH = 'images/'
RESIZE_VALUE = (256, 256)
INITIAL_DATASET = 'clouds_0.csv'
ENRICHED_DATASET = 'clouds_2.csv'
ORIGINAL_IMAGE_WIDTH = 2100
ORIGINAL_IMAGE_HEIGHT = 1400
OUTPUT_SHAPE = (8, 8)
BATCH_SIZE = 128
EPOCH = 20
NB_CLASSES = 4
EFFICIENT_NET_INPUT_SHAPE = (256, 256, 3)
LEARNING_RATE = 1e-3
TRAIN_SPLIT_SIZE = 0.7
HIST_FILE = 'training_history.csv'
MODEL_FILE = 'model.h5'
NB_IMAGES_VISU = 4
NB_COLUMN_VISU = 2

@st.cache_data
def load_initial_data():
    data = pd.read_csv(INITIAL_DATASET)
    return data

@st.cache_data
def load_enriched_data():
    data = pd.read_csv(ENRICHED_DATASET)
    return data

@st.cache_data
def load_hist_data():
    data = pd.read_csv(HIST_FILE)
    return data

@st.cache_resource()
def load_model():
    return tf.keras.models.load_model(MODEL_FILE, custom_objects={'global_loss': global_loss, 'mean_iou': mean_iou})

###########################################
# Initialisation des variables de session #
###########################################

if 'dataframe_11' not in st.session_state:
    st.session_state.dataframe_11 = None
    dataframe_11 = None
else:
    dataframe_11 = st.session_state.dataframe_11

if 'dataframe_21' not in st.session_state:
    st.session_state.dataframe_21 = None
    dataframe_21 = None
else:
    dataframe_21 = st.session_state.dataframe_21

if 'dataframe_31' not in st.session_state:
    st.session_state.dataframe_31 = None
    dataframe_31 = None
else:
    dataframe_31 = st.session_state.dataframe_31

if 'isCharger3' not in st.session_state:
    st.session_state.isCharger3 = None
    isCharger3 = None
else:
    isCharger3 = st.session_state.isCharger3

st.title(":two: Exploration et visualisation des données")
display_info("Cette phase consiste à explorer et visualiser le jeu de données de départ en conjonction avec les images brutes (non réduites).")

#########################################
# Chargement du jeu de données initiale #
#########################################
if st.session_state.dataframe_11 is None:
    st.session_state.dataframe_11 = load_initial_data()
    dataframe_11 = st.session_state.dataframe_11

#################################
# Aperçu du jeu de données brut #
#################################

st.markdown("### 1. Aperçu du jeu de données enrichi")

st.dataframe(dataframe_11.head(10))

# Informations sur le dataset
if st.checkbox("Afficher les informations", key='info_1'):
    info_buffer = io.StringIO()
    dataframe_11.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    st.text(info_output)

    st.info(
        "Les 10 premières observations du jeu de données sont affichées en guise d'aperçu.\
        Afin de réduire la taille du fichier de soumission, l’encodage des segments d’une \
        image (EncodedPixels) a été réalisé à l’aide de la méthode Run-Length Encoding (RLE).\
        \n\nNous constatons que sur les 22184 observations enregistrées dans le Dataset, seulement 11836 seront à priori exploitables.\
        Les étapes suivantes vont consisterà enrichir le Dataset à l'aide de variables explicatives supplémentaires.", icon="ℹ️"
    )

# ##############################
# # Enrichissement des données #
# ##############################

st.markdown("### 2. Enrichissement des données")
display_info("L'objectif de cette étape consiste à rendre le jeu de données exploitable. Les tâches suivantes seront effectuées :")

bullet_points = [
    "_Attribution d'un identifiant unique à chaque observation_",
    "_Estimation de la superficie des segments d'une image_",
    "_Ajout d'une Bounding box autour des segments_",
    "_Données statistiques sur les zones nuageuses_",
    "_Estimation du degré de compacité des segments_",
    "_Encodage de la classe de chaque observation_"
]
display_info_list_items(bullet_points)

if st.button('Charger'):

    if st.session_state.dataframe_21 is None:
        st.session_state.dataframe_21 = load_enriched_data()
        dataframe_21 = st.session_state.dataframe_21

    if st.session_state.isCharger3 != True:
        st.session_state.isCharger3 = True
        isCharger3 = True

if isCharger3:

    st.dataframe(dataframe_21.head(10))

    # Informations sur le dataset
    if st.checkbox("Afficher les informations", key='info_2'):


        tab_a1, tab_a2 = st.tabs(['Structure du Dataset', 'Nomenclature des champs'])
        with tab_a1:
            info_buffer = io.StringIO()
            dataframe_21.info(buf=info_buffer)
            info_output = info_buffer.getvalue()
            st.text(info_output)

        with tab_a2:
            champs = [
                "**Image_Label** : Identifiant unique d'une observation",
                "**FileName** : Nom du fichier image",
                "**EncodedPixels** : Encodage RLE du segment",
                "**EncodedPixelsCount** : Nombre de pixels dans un segment",
                "**segmentCroppingRate** : Pourcentage de rognage d'un segment par une barre noire",
                "**XMOY, YMOY, W, H** : Coordonnées de la Bounding Box",
                "**ResizedMaskPixelsCount** : Nombre de pixels dans le segment d'une image réduite",
                "**BoundingBoxPixelsCount** : Nombre de pixels dans le segment d'une image réduite",
                "**BoxMaskGap** : Ecart entre la superficie réelle d'un segment et la superficie de la Bounding Box",
                "**Label** : Nom de la classe de nuage (Fish, Flower, Gravel, Sugar)",
                "**Class_<Label>** : Encodage one hot da la classification des images"
            ]
            display_info_list_items(champs)

############################################
# Séparation du jeu de données des données #
############################################

#if isCharger3:

    # # On retire 100 images mono-class du jeu d'entraînement et de test que le système ne voit pas durant l'entraînement
    # tmp = dataframe_21[dataframe_21['EncodedPixels'] != -1]
    # value_counts = tmp['FileName'].value_counts()
    # mono_class_items = tmp[tmp['FileName'].map(value_counts) == 1].head(100).FileName.tolist()
    # df_mono_class_unseen = tmp[tmp['FileName'].isin(mono_class_items)]
    # df_mono_class_unseen.reset_index(drop=True, inplace=True)

    # df_train_test = dataframe_21[~dataframe_21['FileName'].isin(mono_class_items)]

    # # Création d'une liste de 100 images mono-class vu par le système pendant l'apprentissage
    # tmp = df_train_test[df_train_test['EncodedPixels'] != -1]
    # value_counts = tmp['FileName'].value_counts()
    # mono_class_items_seen = tmp[tmp['FileName'].map(value_counts) == 1].head(100).FileName.tolist()
    # df_mono_class_seen = tmp[tmp['FileName'].isin(mono_class_items_seen)]
    # df_mono_class_seen.reset_index(drop=True, inplace=True)

########################
# Définition du modèle #
########################

st.markdown("### 3. Définition du modèle")

# if isCharger3:

#     # Backbone
#     efficientNet = EfficientNetB0(include_top=False, input_shape=EFFICIENT_NET_INPUT_SHAPE, weights="imagenet")

#     # Freeze the blackbone
#     for layer in efficientNet.layers:
#         layer.trainable = False

#     model = Sequential()

#     # Feature extraction
#     model.add(efficientNet) 
#     model.add(Reshape([-1, 1280]))

#     # Regression
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(5 + NB_CLASSES))

#     model.summary(print_fn=lambda x: st.text(x))
if isCharger3:
    st.image(GRAPHS_PATH + 'graph_16.png')

###########################################
# Chargement de la courbe d'apprentissage #
###########################################

    st.markdown("### 4. Affichage de la courbe d'apprentissage")
    st.image(GRAPHS_PATH + 'graph_17.png')
    if st.session_state.dataframe_31 is None:
        st.session_state.dataframe_31 = load_hist_data()
        dataframe_31 = st.session_state.dataframe_31

    fig3, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

    axes[0].set_title('Loss')
    axes[0].plot(dataframe_31['loss'])
    axes[0].plot(dataframe_31['val_loss'])
    axes[0].legend(['Train', 'Test'], loc='upper left')

    axes[1].set_title('Mean IOU')
    axes[1].plot(dataframe_31['mean_iou'])
    axes[1].plot(dataframe_31['val_mean_iou'])
    axes[1].legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    st.pyplot(fig3)

    st.info(
        "Pas top du tout :/", icon="ℹ️"
    )

##############
# Prédiction #
##############

    st.markdown("### 5. Prédiction")

    # Chargement du modèle depuis la sauvegrade model.h5
    if os.path.exists(MODEL_FILE):
        model = load_model()

    if st.button('Prédire'):
        fName = '002be4f.jpg'
        imLabel = '002be4f.jpg_Flower'
        img = tf.io.read_file(RESIZED_PATH + fName)
        img = tf.image.decode_png(img, channels=3)

        fig4, axes = plt.subplots(1, 2, figsize=(12, 7))
        show_prediction(img , model, axes[0], threshold=0.6)
        show_ground(imLabel, axes[1], dataframe_21, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT, IMAGES_PATH, hide_axis=True, show_mask=True)
                    
        plt.tight_layout()
        st.pyplot(fig4)

