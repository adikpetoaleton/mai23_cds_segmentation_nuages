import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import random
import io
from utils import *

st.set_page_config(layout='centered')

@st.cache_data
def load_Initial_data():
    data = pd.read_csv('clouds_0.csv')
    return data

@st.cache_data
def load_cleaned_data():
    data = pd.read_csv('clouds_1.csv')
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
    st.session_state.dataframe_1 = load_Initial_data()
    dataframe_1 = st.session_state.dataframe_1

#################################
# Aperçu du jeu de données brut #
#################################

st.markdown("### 1. Aperçu du jeu de données brut")

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
        \nNous constatons que sur les 22184 observations enregistrées dans le Dataset, seulement 11836 seront à priori exploitables.\
        Les étapes suivantes vont consister à nettoyer le jeu de données en retirant les champs vides puis à enrichir le Dataset à \
        l'aide de variables explicatives supplémentaires.", icon="ℹ️"
    )

#########################
# Nettoyage des données #
#########################

st.markdown("### 2. Nettoyage et enrichissement des données")
display_info("L'objectif de cette étape consiste à rendre le jeu de données exploitable. Les tâches suivantes seront effectuées :")

bullet_points = [
    "_Suppression des lignes vides_",
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
        st.session_state.dataframe_2 = load_cleaned_data()
        dataframe_2 = st.session_state.dataframe_2

    if st.session_state.isCharger != True:
        st.session_state.isCharger = True
        isCharger = True

if isCharger:
    st.dataframe(dataframe_2.head(10))

if isCharger:
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
                "**ImageId** : Identifiant unique d'uneobservation",
                "**FileId** : Identifiant unique d'un fichier image",
                "**EncodedPixels** : Encodage RLE du segment",
                "**EncodedPixelsCount** : Nombre de pixels dans un segment",
                "**MeanPixelsCloud** : Moyenne des niveaux des pixels dans une zone nuageuse",
                "**StdPixelsCloud** : Ecart-type des niveaux des pixels dans une zone nuageuse",
                "**CountPixelsHiddenArea** : Nombre de pixels dans une barre noire",
                "**RatioHiddenArea** : Taux d'occupation d'une barre noire par rapport à l'ensemble de l'image",
                "**CloudRatio** : Taux d'occupation d'une zone nuageuse par rapport à l'ensemble de l'image",
                "**X, Y, W, H** : Coordonnées de la Bounding Box",
                "**ResizedMaskPixelsCount** : Nombre de pixels dans le segment d'une image réduite",
                "**BoundingBoxPixelsCount** : Nombre de pixels dans le segment d'une image non réduite",
                "**BoxMaskGap** : Ecart entre la superficie réelle d'un segment et sa Bounding Box",
                "**Label** : Nom de la classe de nuage",
                "**Class** : Code de la classe de nuage"
            ]
            display_info_list_items(champs)



###########################
# Exploration des données #
###########################

if isCharger:

    st.markdown("### 3. Exploration des données")

    class_per_image = dataframe_2.groupby('FileId').agg({'Class': 'count'}).rename({'Class':'Occurence'}, axis=1)
    dist1 = pd.DataFrame(dataframe_2['Class'].value_counts(normalize=True))
    dist1.reset_index(drop=False, inplace=True)
    dist1 = dist1.rename({'Class':'Ratio', 'index': 'Class'}, axis=1)
    dist1 = dist1.replace(to_replace=[0, 1, 2, 3], value=['Fish', 'Flower', 'Gravel', 'Sugar'])

    dist2 = pd.DataFrame(class_per_image['Occurence'].value_counts())
    dist2.reset_index(drop=False, inplace=True)
    dist2.sort_values(by='Occurence', ascending=True, inplace=True)
    dist2 = dist2.replace(to_replace=[1, 2, 3, 4], value=['1 Label(s)', '2 Label(s)', '3 Label(s)', '4 Label(s)'])

    ##############################
    # Affichage des statistiques #
    ##############################

    # Stats n°1
    st.markdown("##### Résumé des statistiques (describe)")
    st.dataframe(dataframe_2.describe())

    # Stats n°2
    st.markdown("##### Autres statistiques")
    tab_b1, tab_b2 = st.tabs(['Distribution des classes de nuage', 'Distribution selon le nombre de classes qui les caractérisent'])
    
    with tab_b1:

        col_b11, col_b12 = st.columns([1, 2], gap='small')
        
        with col_b11:
            st.dataframe(dist1)

        with col_b12:
            fig2, ax2 = plt.subplots()
            ax2.pie(
                x = dist1['proportion'], 
                labels = dist1['Ratio'],
                autopct = lambda x: str(round(x, 2)) + '%',
                pctdistance = 0.7, 
                labeldistance = 1.1,
                shadow = False,
                textprops = {'fontsize': 8}
            )
            st.pyplot(fig2)

    with tab_b2:
        col_b21, col_b22 = st.columns([1, 2], gap='small')
        with col_b21:
            st.dataframe(dist2)
        with col_b22:
            fig1, ax1 = plt.subplots()
            sns.set_theme()
            sns.barplot(x='Occurence', y='count', data=dist2)
            st.pyplot(fig1)

    # Stats n°3
    st.markdown("##### Distribution de l'étendu des nuages par classe de nuage")

    g = sns.FacetGrid(dataframe_2, col='Label', height=4)
    g.map_dataframe(sns.histplot, 'CloudRatio', kde=True, bins=15)
    st.pyplot(plt)

    display_info("Commentaires ici...")

    # Stats n°4
    st.markdown("##### Quelques distributions...")

    exp1 = st.expander("Distribution du taux de couverture des nuages (par rapport à l'image entière) par classe de nuage", expanded=True)
    with exp1:
        sns.set(style='whitegrid')
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Label', y='CloudRatio', data=dataframe_2, width=0.5)
        sns.despine(left=True)
        st.pyplot(fig3)

    exp2 = st.expander("Distribution de la moyenne des niveaux de pixel d'une zone nuageuse par classe de nuage")
    with exp2:
        sns.set(style='whitegrid')
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Label', y='MeanPixelsCloud', data=dataframe_2, width=0.5)
        sns.despine(left=True)
        st.pyplot(fig4)

    exp3 = st.expander("Distribution de l'écart-type des niveaux de pixel d'une zone nuageuse par classe de nuage")
    with exp3:
        sns.set(style='whitegrid')
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Label', y='StdPixelsCloud', data=dataframe_2, width=0.5)
        sns.despine(left=True)
        st.pyplot(fig5)
    
    ############################
    # Visualisation des images #
    ############################

    st.markdown("### 4. Visualisation des images")
    IMG_PATH = 'images/'
    
    if st.checkbox("Afficher", key='xxx'):
        with st.spinner("Veuillez patienter"):
            exp1 = st.expander("Visualiser des images multi-classes")
            with exp1:
                ImageIds = ['002be4f_0', '002be4f_1', '002be4f_3']
                showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)

            exp2 = st.expander("Visualiser des images decrites par des segments compactes")
            with exp2:
                ImageIds = ['c22cfa7_3', 'f128b90_1', '6906aa0_3']
                showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)

            exp3 = st.expander("Visualiser des images mono-classe decrites par des segments disjoints")
            with exp3:
                ImageIds = ['db36366_2', '5717e63_3', '4a7b6e3_3', '37e8349_0', 'ae7a06d_3', 'c465c2e_3']
                showImages(ImageIds, 2, 3, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=True)

            exp4 = st.expander("Visualiser des images présentant une zone cachée significative")
            with exp4:
                ImageIds = ['24884e7_0', 'f32724b_0', '3b9a092_0', '5265e81_0', '400a38d_0', 'a2dc5c0_0', '17fe76e_0', '838cd7a_0', 'fd5aa5d_0', '42ac1b7_0', '171e62f_0', '06e5dd6_0']
                showImages(ImageIds, 3, 4, dataframe_2, 2100, 1400, IMG_PATH, hide_axis=True, show_mask=False)


