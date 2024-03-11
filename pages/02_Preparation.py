import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import random
import io
from utils import *

st.set_page_config(page_title="Cloud manipulation", page_icon=":three:", layout='centered')

@st.cache_data
def load_cleaned_data():
    data = pd.read_csv('clouds_1.csv')
    return data

###########################################
# Initialisation des variables de session #
###########################################

if 'dataframe_4' not in st.session_state:
    st.session_state.dataframe_4 = None
    dataframe_4 = None
else:
    dataframe_4 = st.session_state.dataframe_4

st.title(":three: Préparation des données en vue de la segmentation")
display_info("Cette phase consiste à retirer les abérrations du jeu de données en vue de la segmentation des images.")

####################################
# Aperçu du jeu de données enrichi #
####################################

st.markdown("### 1. Aperçu du jeu de données enrichi précédemment")

# Chargement du jeu de données enrichi
if st.session_state.dataframe_4 is None:
    st.session_state.dataframe_4 = load_cleaned_data()
    dataframe_4 = st.session_state.dataframe_4

st.dataframe(dataframe_4.head(10))

# Informations sur le dataset
if st.checkbox("Afficher les informations", key='info_1'):
    info_buffer = io.StringIO()
    dataframe_4.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    st.text(info_output)

    st.info(
        "Rappelons qu'en réalité, ce jeu de données est constitué de 11836 observations potentiellement exploitables.\
        Une limitation de la plateforme GitHub nous oblige à limiter la taille du fichier correspondant à 100Mo.", icon="ℹ️"
    )

##############################
# Filtrage des observations  #
##############################

st.markdown("### 2. Filtrage des observations")
display_info("A partir de cet instant, nous allons procéder à 2 étapes de filtrage des observations à savoir la purge des images significativement \
altérées par une barre noire, puis celle des images dont les segments sont disjoints par nature.")

st.markdown("##### Filtrage des observations masquées par une barre noire de façon significative")

tab_a1, tab_a2 = st.tabs(['Observations masquées', 'Jeu de données filtré'])

# Identification des observations masquées
with tab_a1:
    df_hidden = dataframe_4[dataframe_4['RatioHiddenArea'] > 0.2][['FileId', 'CountPixelsHiddenArea', 'RatioHiddenArea']].groupby('FileId').agg({'CountPixelsHiddenArea': lambda x: x.mode().iloc[0], 'RatioHiddenArea': lambda x: x.mode().iloc[0]}).reset_index()
    df_hidden = df_hidden.sort_values(by='RatioHiddenArea', ascending=False)
    st.dataframe(df_hidden)


    # Visualisation des observations masquées

    if st.checkbox("Visualisation des observations concernées", key='info_5'):
        with st.spinner("Veuillez patienter"):

            ImageIds = ['24884e7_0', '3b9a092_0', '5265e81_0', '400a38d_0', '17fe76e_0', '42ac1b7_0', '171e62f_0', '06e5dd6_0', '076de5e_0']
            showImages(ImageIds, 3, 3, dataframe_4, 2100, 1400, './images/', hide_axis=True, show_mask=False)

# Filtrage des observations masquées
with tab_a2:
    dataframe_4 = dataframe_4[dataframe_4['FileId'].apply(lambda x: all(exclude not in x for exclude in df_hidden['FileId'].tolist()))]
    st.dataframe(dataframe_4.head(10))

    # Informations sur le jeu de données
    if st.checkbox("Afficher les informations", key='info_2'):
        info_buffer = io.StringIO()
        dataframe_4.info(buf=info_buffer)
        info_output = info_buffer.getvalue()
        st.text(info_output)

st.info(
    "Le jeu de données contient une dizaine d'images abérrantes dans ce sens que la barre noire occupe plus de 20% de l'ensemble de l'image. \
    Il s'agit ici d'un seuil arbitraire qui n'impacte pas du tout le volume de données.", icon="ℹ️"
)

# Filtrage des observations avec segments disjoints

st.markdown("##### Filtrage des observations avec segments disjoints")
st.info(
    "En appliquant ce filtre, nous retirons du jeu de données l'ensemble des observations \
    pour lesquelles les segments de zone nuageuse ne délimitent pas une étendue continue.\
    Il peut s'agir de véritable zone nuageuse disjointe et de même catégorie ou alors \
    d'une zone traversée par une barre noire.\
    Nous considérons raisonnablement qu'un écart de plus de 9% entre la superficie d'un segment et celle\
    de la Bounding Box peut être considéré comme un cas de segment disjoint.", icon="ℹ️"
)

if st.checkbox("Filtrer les observations", key='info_3'):

    GAP_LIMIT_LOW = 0
    GAP_LIMIT_HIGH = 0.09

    dataframe_4 = dataframe_4[(dataframe_4['BoxMaskGap'] >= GAP_LIMIT_LOW) & (dataframe_4['BoxMaskGap'] < GAP_LIMIT_HIGH)]
    
    st.dataframe(dataframe_4)
    
    if st.checkbox("Afficher les informations", key='info_4'):
        info_buffer = io.StringIO()
        dataframe_4.info(buf=info_buffer)
        info_output = info_buffer.getvalue()
        st.text(info_output)

##################################
# Visualisation des bounding Box #
##################################

st.markdown("### 3. Visualisation de quelques observations candidates à la segmentation")

if st.button('Afficher / Rafraîchir'):

    with st.spinner("Veuillez patienter"):
        ImageIds = random.sample(dataframe_4['ImageId'].unique().tolist(), 3)
        showImages(ImageIds, 1, 3, dataframe_4, 2100, 1400, 'images/', hide_axis=False, show_mask=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 20), layout='constrained')
        for axe, img_id in zip(axes.flat, ImageIds):
            x = dataframe_4[dataframe_4['ImageId'] == img_id].X
            y = dataframe_4[dataframe_4['ImageId'] == img_id].Y
            w = dataframe_4[dataframe_4['ImageId'] == img_id].W
            h = dataframe_4[dataframe_4['ImageId'] == img_id].H
            displayBoundingBox(img_id, axe, x, y, w, h)
        st.pyplot(fig)

    st.info(
        "Sur la rangée supérieure, nous observons un échantillon aléatoire d'images à leur taille d'origine sur laquelle nous superposons le segment de nuage, \
        alors que la rangée inférieure représente ces mêmes images mais réduites de moitié, et sur les quelles nous superposons une Bounding Box en vue de la phase de segmentation.", icon="ℹ️"
    )
