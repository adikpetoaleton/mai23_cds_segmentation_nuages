import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import custom_object_scope

from utils import *
import io

st.set_page_config(
    layout='centered'
)

@st.cache_data
def load_data():
    data = pd.read_csv('clouds_2.csv')
    return data

# Initialisation des variables de session
if 'dataframe_1' not in st.session_state:
    st.session_state['dataframe_1'] = None

if 'dataframe_2' not in st.session_state:
    st.session_state['dataframe_2'] = None

if 'dataframe_3' not in st.session_state:
    st.session_state['dataframe_3'] = None

if 'issampled' not in st.session_state:
    st.session_state['issampled'] = False

st.title("Préparation des données")
display_info("Cette phase consiste à préparer le jeu de données en conjonction avec les images\
             réduites de sorte qu'il soit exploitable pour l'étape de modélisation.")

# Chargement du jeu de données enrichi
if 'dataframe_1' in st.session_state:
    if st.session_state['dataframe_1'] is None:
        st.session_state['dataframe_1'] = load_data()
    df_1 = st.session_state['dataframe_1']

# ####################################
# # Aperçu du jeu de données enrichi #
# ####################################

st.markdown("### 1. Aperçu du jeu de données enrichi")
st.dataframe(df_1.head())

# Informations sur le dataset
if st.checkbox("Afficher les informations", key=1):
    info_buffer = io.StringIO()
    df_1.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    st.text(info_output)

###########################
# Préparation des données #
###########################

st.markdown("### 2. Préparation des données")

# Echantillonage des images avec classes non-disjointes
st.markdown("##### 2.1 Echantillonage des images avec classes non-disjointes")
display_info("Cette étape consiste à sélectionner les images qui ne possèdent qu'une seule classe\
                et pour lesquelles le segment n'est pas discontinu.")

GAP_LIMIT_LOW = 0
GAP_LIMIT_HIGH = 0.09 # (< 0.09 -> 9%)
df_2 = df_1[(df_1['BoxMaskGap'] >= GAP_LIMIT_LOW) & (df_1['BoxMaskGap'] < GAP_LIMIT_HIGH)]

if 'issampled' in st.session_state:
    st.session_state['issampled'] = True

if 'dataframe_2' in st.session_state:
    st.session_state['dataframe_2'] = df_2

st.dataframe(df_2.head())

if st.session_state['issampled']:

    if st.checkbox("Afficher les informations", key=2) :
        info_buffer = io.StringIO()
        df_2.info(buf=info_buffer)
        info_output = info_buffer.getvalue()
        st.text(info_output)

    # Standardisation des données
    st.markdown("##### 2.2 Standardisation des données")
    display_info("Cette étape consiste à standardiser les coordonées de la Bounding box.")

    df_3 = df_2

    df_3['X'] = df_3['X'] / 525
    df_3['Y'] = df_3['Y'] / 350
    df_3['W'] = df_3['W'] / 525
    df_3['H'] = df_3['H'] / 350

    # Marquage des images milti-classes
    df_3 = markDuplicate(df_3, 'FileId', 'Class')

    if 'dataframe_3' in st.session_state:
        st.session_state['dataframe_3'] = df_3
    
    st.dataframe(df_3.head())

######################
# Création du modèle #
######################

    st.markdown("### 3. Création du modèle")

    # Chargement du modèle EfficientNetB0 (Ensemble de convolution)
    efficientNet = EfficientNetB0(include_top=False, input_shape=(350, 525, 3))

    # Freeze the blackbone
    for layer in efficientNet.layers:
        layer.trainable = False

    # Ajout des couches Dense pour la partie régression et classification
    input_model = Input(shape=[350, 525, 3])
    feature = efficientNet(input_model)
    x = GlobalAveragePooling2D()(feature)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    object_prob = Dense(1, activation='sigmoid')(x)
    regression = Dense(4, activation='linear')(x)
    output_model = tf.concat([object_prob, regression], axis=-1)

    model = Model(inputs=input_model, outputs=output_model)
    
    if st.checkbox("Afficher les informations du modèle"):
        st.write(model.summary())

#     # Définition de la fontion perte
