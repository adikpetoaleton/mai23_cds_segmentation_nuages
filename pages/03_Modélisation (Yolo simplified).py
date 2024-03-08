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
tf.keras.backend.clear_session()

st.set_page_config(layout='centered')

#########################################
# Initialisation des variables globales #
#########################################

RESIZED_PATH = 'images/resized/'
IMAGES_PATH = 'images/'
RESIZE_VALUE = (256, 256)
INITIAL_DATASET = 'clouds_0_small.csv'
ENRICHED_DATASET = 'clouds_1_small.csv'
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

###########################################
# Initialisation des variables de session #
###########################################

if 'dataframe_1' not in st.session_state:
    st.session_state.dataframe_1 = None
    dataframe_1 = None
else:
    dataframe_1 = st.session_state.dataframe_1

if 'dataframe_2' not in st.session_state:
    st.session_state.dataframe_2 = None
    dataframe_2 = None
else:
    dataframe_2 = st.session_state.dataframe_2

if 'isCharger' not in st.session_state:
    st.session_state.isCharger = None
    isCharger = None
else:
    isCharger = st.session_state.isCharger

st.title("Exploration et visualisation des données")
display_info("Cette phase consiste à explorer et visualiser le jeu de données de départ en conjonction avec les images brutes (non réduites).")

#########################################
# Chargement du jeu de données initiale #
#########################################
if st.session_state.dataframe_1 is None:
    st.session_state.dataframe_1 = load_initial_data()
    dataframe_1 = st.session_state.dataframe_1

#################################
# Aperçu du jeu de données brut #
#################################

st.markdown("### 1. Aperçu du jeu de données enrichi")

st.dataframe(dataframe_1.head(10))

# Informations sur le dataset
if st.checkbox("Afficher les informations", key='info_1'):
    info_buffer = io.StringIO()
    dataframe_1.info(buf=info_buffer)
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

    if st.session_state.dataframe_2 is None:
        st.session_state.dataframe_2 = load_enriched_data()
        dataframe_2 = st.session_state.dataframe_2

    if st.session_state.isCharger != True:
        st.session_state.isCharger = True
        isCharger = True

if isCharger:

    st.dataframe(dataframe_2.head(10))

    # Informations sur le dataset
    if st.checkbox("Afficher les informations", key='info_2'):


        tab_a1, tab_a2 = st.tabs(['Structure du Dataset', 'Nomenclature des champs'])
        with tab_a1:
            info_buffer = io.StringIO()
            dataframe_2.info(buf=info_buffer)
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

if isCharger:

    # On retire 100 images mono-class du jeu d'entraînement et de test que le système ne voit pas durant l'entraînement
    tmp = dataframe_2[dataframe_2['EncodedPixels'] != -1]
    value_counts = tmp['FileName'].value_counts()
    mono_class_items = tmp[tmp['FileName'].map(value_counts) == 1].head(100).FileName.tolist()
    df_mono_class_unseen = tmp[tmp['FileName'].isin(mono_class_items)]
    df_mono_class_unseen.reset_index(drop=True, inplace=True)

    df_train_test = dataframe_2[~dataframe_2['FileName'].isin(mono_class_items)]

    # Création d'une liste de 100 images mono-class vu par le système pendant l'apprentissage
    tmp = df_train_test[df_train_test['EncodedPixels'] != -1]
    value_counts = tmp['FileName'].value_counts()
    mono_class_items_seen = tmp[tmp['FileName'].map(value_counts) == 1].head(100).FileName.tolist()
    df_mono_class_seen = tmp[tmp['FileName'].isin(mono_class_items_seen)]
    df_mono_class_seen.reset_index(drop=True, inplace=True)

########################
# Définition du modèle #
########################

st.markdown("### 3. Définition du modèle")
display_info("à compléter")

if isCharger:

    # Backbone
    efficientNet = EfficientNetB0(include_top=False, input_shape=EFFICIENT_NET_INPUT_SHAPE, weights="imagenet")

    # Freeze the blackbone
    for layer in efficientNet.layers:
        layer.trainable = False

    model = Sequential()

    # Feature extraction
    model.add(efficientNet) 
    model.add(Reshape([-1, 1280]))

    # Regression
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5 + NB_CLASSES))

    model.summary()





# if isCharger:

#     st.markdown("### 3. Exploration des données")

#     class_per_image = dataframe_2.groupby('FileId').agg({'Class': 'count'}).rename({'Class':'Occurence'}, axis=1)
#     dist1 = pd.DataFrame(dataframe_2['Class'].value_counts(normalize=True))
#     dist1.reset_index(drop=False, inplace=True)
#     dist1 = dist1.rename({'Class':'Ratio', 'index': 'Class'}, axis=1)
#     dist1 = dist1.replace(to_replace=[0, 1, 2, 3], value=['Fish', 'Flower', 'Gravel', 'Sugar'])

#     dist2 = pd.DataFrame(class_per_image['Occurence'].value_counts())
#     dist2.reset_index(drop=False, inplace=True)
#     dist2.sort_values(by='Occurence', ascending=True, inplace=True)
#     dist2 = dist2.replace(to_replace=[1, 2, 3, 4], value=['1 Label', '2 Labels', '3 Labels', '4 Labels'])

#     ##############################
#     # Affichage des statistiques #
#     ##############################

#     # Stats n°1
#     st.markdown("##### Résumé des statistiques (describe)")
#     st.dataframe(dataframe_2.describe())

#     # Stats n°2
#     st.markdown("##### Autres statistiques")
#     tab_b1, tab_b2 = st.tabs(['Distribution des classes de nuage', 'Distribution selon le nombre de classes qui les caractérisent'])
    
#     with tab_b1:

#         col_b11, col_b12 = st.columns([1, 2], gap='small')
        
#         with col_b11:
#             st.dataframe(dist1)

#         with col_b12:
#             fig2, ax2 = plt.subplots()
#             ax2.pie(
#                 x = dist1['proportion'], 
#                 labels = dist1['Ratio'],
#                 autopct = lambda x: str(round(x, 2)) + '%',
#                 pctdistance = 0.7, 
#                 labeldistance = 1.1,
#                 shadow = False,
#                 textprops = {'fontsize': 8}
#             )
#             st.pyplot(fig2)

#     with tab_b2:
#         col_b21, col_b22 = st.columns([1, 2], gap='small')
#         with col_b21:
#             st.dataframe(dist2)
#         with col_b22:
#             fig1, ax1 = plt.subplots()
#             sns.set_theme()
#             sns.barplot(x='Occurence', y='count', data=dist2)
#             st.pyplot(fig1)

#     # Stats n°3
#     st.markdown("##### Distribution de l'étendu des nuages par classe de nuage")

#     g = sns.FacetGrid(dataframe_2, col='Label', height=4)
#     g.map_dataframe(sns.histplot, 'CloudRatio', kde=True, bins=15)
#     st.pyplot(plt)

#     display_info("Commentaires ici...")

#     # Stats n°4
#     st.markdown("##### Quelques distributions...")

#     exp1 = st.expander("Distribution du taux de couverture des nuages (par rapport à l'image entière) par classe de nuage", expanded=True)
#     with exp1:
#         sns.set(style='whitegrid')
#         fig3, ax3 = plt.subplots(figsize=(8, 6))
#         sns.boxplot(x='Label', y='CloudRatio', data=dataframe_2, width=0.5)
#         sns.despine(left=True)
#         st.pyplot(fig3)

#     exp2 = st.expander("Distribution de la moyenne des niveaux de pixel d'une zone nuageuse par classe de nuage")
#     with exp2:
#         sns.set(style='whitegrid')
#         fig4, ax4 = plt.subplots(figsize=(8, 6))
#         sns.boxplot(x='Label', y='MeanPixelsCloud', data=dataframe_2, width=0.5)
#         sns.despine(left=True)
#         st.pyplot(fig4)

#     exp3 = st.expander("Distribution de l'écart-type des niveaux de pixel d'une zone nuageuse par classe de nuage")
#     with exp3:
#         sns.set(style='whitegrid')
#         fig5, ax5 = plt.subplots(figsize=(8, 6))
#         sns.boxplot(x='Label', y='StdPixelsCloud', data=dataframe_2, width=0.5)
#         sns.despine(left=True)
#         st.pyplot(fig5)
    
#     ############################
#     # Visualisation des images #
#     ############################

#     st.markdown("### 4. Visualisation des images")
#     IMG_PATH = 'images/'
    
#     if st.checkbox("Afficher", key='xxx'):

#         with st.spinner("Veuillez patienter"):

#             exp1 = st.expander("Visualiser des images multi-classes", expanded=True)
#             with exp1:
#                 ImageIds = ['002be4f_0', '002be4f_1', '002be4f_3']
#                 showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)

#             exp2 = st.expander("Visualiser des images decrites par des segments compactes")
#             with exp2:
#                 ImageIds = ['659c0e7_0', '2b335f2_1', '6906aa0_3']
#                 showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)

#             exp3 = st.expander("Visualiser des images mono-classe decrites par des segments disjoints")
#             with exp3:
#                 ImageIds = ['5717e63_0', '4a7b6e3_3', '37e8349_0']
#                 showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)

#             exp4 = st.expander("Visualiser des images présentant une zone cachée significative")
#             with exp4:
#                 ImageIds = ['f32724b_0', '17fe76e_0', '06e5dd6_0']
#                 showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=False)
            
#             # exp1 = st.expander("Visualiser des images multi-classes")
#             # with exp1:
#             #     ImageIds = random.sample(dataframe_2['ImageId'].unique().tolist(), 9)
#             #     showImages(ImageIds, 3, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)


