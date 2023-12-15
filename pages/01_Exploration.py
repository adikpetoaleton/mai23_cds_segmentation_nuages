import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import random
import io
from utils import *

st.set_page_config(
    layout='centered'
)

@st.cache_data
def load_data():
    data = pd.read_csv('clouds_1_xsmall.csv')
    return data

@st.cache_data
def load_data_1():
    data = pd.read_csv('clouds_1_xsmall.csv')
    return data

# Initialisation des variables de session
if 'iscleaned' not in st.session_state:
    st.session_state['iscleaned'] = False

if 'dataframe_1' not in st.session_state:
    st.session_state['dataframe_1'] = None

if 'dataframe_2' not in st.session_state:
    st.session_state['dataframe_2'] = None

st.title("Exploration et visualisation des données")
display_info("Cette phase consiste à explorer et visualiser le jeu de données de départ\
             en conjonction avec les images brutes (non réduites).")

# Chargement du jeu de données initiale
if 'dataframe_1' in st.session_state:
    if st.session_state['dataframe_1'] is None:
        st.session_state['dataframe_1'] = load_data()
    df_1 = st.session_state['dataframe_1']

#################################
# Aperçu du jeu de données brut #
#################################

st.markdown("### 1. Aperçu du jeu de données brut")
display_info("Les 10 premières observations du jeu de données sont affichées en guise d'aperçu.")

st.dataframe(df_1.head(10))

# Informations sur le dataset
if st.checkbox("Afficher les informations", key='info_1') :
    info_buffer = io.StringIO()
    df_1.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    st.text(info_output)

#########################
# Nettoyage des données #
#########################

st.markdown("### 2. Nettoyage des données")
display_info("L'objectif de cette étape consiste à supprimer les observations qui possèdent\
          des champs nuls et à rendre les variables du jeu de données exploitables.")

# Chargement des données pré-traitées (c'est plus rapide)
if st.checkbox("Données pré-traitées (rapide)"):

    if st.button('Charger') or st.session_state['iscleaned']:

        if 'dataframe_2' in st.session_state:
            st.session_state['dataframe_2'] = load_data_1()
            df_2 = st.session_state['dataframe_2']

        if 'iscleaned' in st.session_state:
            st.session_state['iscleaned'] = True

        st.dataframe(df_2.head())

        # Informations sur le dataset
        if st.checkbox("Afficher les informations", key='info_2') :
            info_buffer = io.StringIO()
            df_2.info(buf=info_buffer)
            info_output = info_buffer.getvalue()
            st.text(info_output)

# On relance la préparation des données   
else:
   if st.button('Démarrer'):
        
        df_2 = df_1.dropna(axis=0, how='any')
        df_2['Label'] = df_2['Image_Label'].apply(lambda label: label.split('_')[1])
        df_2['FileId'] = df_2['Image_Label'].apply(lambda label: label.split('.')[0])
        df_2['EncodedPixels_array'] = df_2['EncodedPixels'].apply(lambda x: np.array(x.split(' ')).astype(int))
        df_2['EncodedPixelsCount'] = df_2['EncodedPixels_array'].apply(lambda x:  np.sum(x.reshape(int(np.size(x)/2), 2)[:, 1]))
        le = LabelEncoder()
        df_2['Class'] = le.fit_transform(df_2['Label'])
        df_2['ImageId'] = df_2.apply(lambda row: row['Image_Label'].split('.')[0] + '_' + str(row['Class']), axis=1)
        df_2 = df_2.drop(labels=['EncodedPixels_array'], axis=1)
        df_2 = df_2.drop(labels=['Image_Label'], axis=1)
        df_2 = df_2.reindex(columns=['ImageId', 'FileId', 'EncodedPixels', 'EncodedPixelsCount', 'Label', 'Class'])
        
        if 'dataframe_2' in st.session_state:
            st.session_state['dataframe_2'] = df_2

        if 'iscleaned' in st.session_state:
            st.session_state['iscleaned'] = True

        st.dataframe(df_2.head())

        # Informations sur le dataset
        if st.checkbox("Afficher les informations", key='info_3') :
            info_buffer = io.StringIO()
            df_2.info(buf=info_buffer)
            info_output = info_buffer.getvalue()
            st.text(info_output)

##############################
# Affichage des Statistiques #
##############################

if st.session_state['iscleaned']:

    st.markdown("### 3. Statistiques")

    if 'dataframe_2' in st.session_state:
        df_2 = st.session_state['dataframe_2']

    class_per_image = df_2.groupby('FileId').agg({'Class': 'count'}).rename({'Class':'Occurence'}, axis=1)
    dist1 = pd.DataFrame(df_2['Class'].value_counts(normalize=True))
    dist1.reset_index(drop=False, inplace=True)
    dist1 = dist1.rename({'Class':'Ratio', 'index': 'Class'}, axis=1)
    dist1 = dist1.replace(to_replace=[0, 1, 2, 3], value=['Fish', 'Flower', 'Gravel', 'Sugar'])

    dist2 = pd.DataFrame(class_per_image['Occurence'].value_counts())
    dist2.reset_index(drop=False, inplace=True)
    dist2.sort_values(by='Occurence', ascending=True, inplace=True)
    dist2 = dist2.replace(to_replace=[1, 2, 3, 4], value=['1 Label(s)', '2 Label(s)', '3 Label(s)', '4 Label(s)'])

    # Affichage des statistiques
    tab_a, tab_b = st.tabs(['Résumé des statistiques (describe)', 'Autres statistiques'])
    with tab_a:
        st.dataframe(df_2.describe())

    with tab_b:
        col1, col2 = st.columns(2, gap='small')
        with col1:
            st.write("_Répartition des classes_")
            st.dataframe(dist1)
        with col2:
            st.write("_Répartition du nombre de classes par image_")
            st.dataframe(dist2)

    # Affichage des différents graphiques
    tab1, tab2 = st.tabs(['Number of images per Occurence', 'Percentage of images per occurence'])
    with tab1:
        fig1, ax1 = plt.subplots()
        sns.set_theme()
        sns.histplot(class_per_image['Occurence'], kde=False, ax=ax1)
        ax1.set_title("Number of images per Occurence")
        st.pyplot(fig1)

    with tab2:
        fig2, ax2 = plt.subplots()
        ax2.pie(
            x = dist2['count'], 
            labels=dist2['Occurence'],
            autopct=lambda x: str(round(x, 2)) + '%',
            pctdistance =0.7, 
            labeldistance=1.1,
            shadow = False,
            textprops={'fontsize': 8}
        )
        ax2.set_title("Percentage of images per occurence", fontsize=8)
        st.pyplot(fig2)

############################
# Visualisation des images #
############################

if st.session_state['iscleaned']:

    st.markdown("### 4. Visualisation des images")
    display_info("Un échantllon aléatoire de 9 images est affiché avec une mise en évidence la segmentation.")
    
    if st.button("Afficher"):

        with st.spinner("Veuillez patienter"):
            img_width = 2100
            img_height = 1400
            cmap = 'viridis'
            alpha = 0.2
            image_path = 'images/'

            #ImageIds = random.sample(df_2['ImageId'].unique().tolist(), 9)
            ImageIds = ['0011165_1','002be4f_3','003994e_3','0091591_2','00b81e1_3','0100a84_2','011ba04_1','015b764_3','017ded1_3']

            fig, axes = plt.subplots(3, 3, figsize=(20, 15), layout='constrained')
            for axe, img_id in zip(axes.flat, ImageIds):
                displayMasks(img_id, axe, df_2, img_width, img_height, cmap, alpha, image_path)
            st.pyplot(fig)
