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
def load_cleaned_data():
    data = pd.read_csv('clouds_1.csv')
    return data

###########################################
# Initialisation des variables de session #
###########################################

if 'dataframe_2' not in st.session_state:
    st.session_state.dataframe_2 = None
    dataframe_2 = None
else:
    dataframe_2 = st.session_state.dataframe_2

st.title("Préparation des données en vue de la segmentation")
display_info("Cette phase consiste à explorer et visualiser le jeu de données de départ en conjonction avec les images brutes (non réduites).")

####################################
# Aperçu du jeu de données enrichi #
####################################

st.markdown("### 1. Aperçu du jeu de données enrichi précédemment")

# Chargement du jeu de données enrichi
if st.session_state.dataframe_2 is None:
    st.session_state.dataframe_2 = load_cleaned_data()
    dataframe_2 = st.session_state.dataframe_2

st.dataframe(dataframe_2.head(10))

# Informations sur le dataset
if st.checkbox("Afficher les informations", key='info_1'):
    info_buffer = io.StringIO()
    dataframe_2.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    st.text(info_output)

    st.markdown("##### Commentaires")
    display_info(
        "To be defined"
    )

##############################
# Filtrage des observations  #
##############################

st.markdown("### 2. Filtrage des observations masquées par une barre noire")

tab_a1, tab_a2 = st.tabs(['Observations masquées', 'Jeu de données filtré'])

# Identification des observations masquées
with tab_a1:
    df_hidden = dataframe_2[dataframe_2['RatioHiddenArea'] > 0.2][['FileId', 'CountPixelsHiddenArea', 'RatioHiddenArea']].groupby('FileId').agg({'CountPixelsHiddenArea': lambda x: x.mode().iloc[0], 'RatioHiddenArea': lambda x: x.mode().iloc[0]}).reset_index()
    df_hidden = df_hidden.sort_values(by='RatioHiddenArea', ascending=False)
    st.dataframe(df_hidden)


    # Visualisation des observations masquées

    if st.checkbox("Visualisation des observations concernées", key='info_5'):
        with st.spinner("Veuillez patienter"):

            ImageIds = ['24884e7_0', '3b9a092_0', '5265e81_0', '400a38d_0', '17fe76e_0', '42ac1b7_0', '171e62f_0', '06e5dd6_0', '076de5e_0']
            showImages(ImageIds, 3, 3, dataframe_2, 2100, 1400, '../images/', hide_axis=True, show_mask=False)

# Filtrage des observations masquées
with tab_a2:
    dataframe_2 = dataframe_2[dataframe_2['FileId'].apply(lambda x: all(exclude not in x for exclude in df_hidden['FileId'].tolist()))]
    st.dataframe(dataframe_2)

    # Informations sur le jeu de données
    if st.checkbox("Afficher les informations", key='info_2'):
        info_buffer = io.StringIO()
        dataframe_2.info(buf=info_buffer)
        info_output = info_buffer.getvalue()
        st.text(info_output)

        st.markdown("##### Commentaires")
        display_info(
            "To be defined"
        )

# Filtrage des observations avec segments disjoints

st.markdown("### 3. Filtrage des observations avec segments disjoints")
display_info(
    "En appliquant ce filtre, nous retirons du jeu de données l'ensemble des observations \
    pour lesquelles les segments de zone nuageuse ne délimitent pas une étendue continue."
)

if st.checkbox("Filtrer", key='info_3'):

    GAP_LIMIT_LOW = 0
    GAP_LIMIT_HIGH = 0.09

    dataframe_2 = dataframe_2[(dataframe_2['BoxMaskGap'] >= GAP_LIMIT_LOW) & (dataframe_2['BoxMaskGap'] < GAP_LIMIT_HIGH)]
    
    st.dataframe(dataframe_2)
    
    if st.checkbox("Afficher les informations", key='info_4'):
        info_buffer = io.StringIO()
        dataframe_2.info(buf=info_buffer)
        info_output = info_buffer.getvalue()
        st.text(info_output)

##################################
# Visualisation des bounding Box #
##################################

st.markdown("### 4. Visualisation des bounding Box")

if st.button('Afficher / Rafraîchir'):
    with st.spinner("Veuillez patienter"):
        ImageIds = random.sample(dataframe_2['ImageId'].unique().tolist(), 3)
        showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, 'images/', hide_axis=False, show_mask=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 20), layout='constrained')
        for axe, img_id in zip(axes.flat, ImageIds):
            x = dataframe_2[dataframe_2['ImageId'] == img_id].X
            y = dataframe_2[dataframe_2['ImageId'] == img_id].Y
            w = dataframe_2[dataframe_2['ImageId'] == img_id].W
            h = dataframe_2[dataframe_2['ImageId'] == img_id].H
            displayBoundingBox(img_id, axe, x, y, w, h)
        st.pyplot(fig)


# ##############################
# # Affichage des statistiques #
# ##############################

# # Stats n°1
# st.markdown("##### Résumé des statistiques (describe)")
# st.dataframe(dataframe_2.describe())

# # Stats n°2
# st.markdown("##### Autres statistiques")
# tab_b1, tab_b2 = st.tabs(['Distribution des classes de nuage', 'Distribution selon le nombre de classes qui les caractérisent'])

# with tab_b1:

#     col_b11, col_b12 = st.columns([1, 2], gap='small')
    
#     with col_b11:
#         st.dataframe(dist1)

#     with col_b12:
#         fig2, ax2 = plt.subplots()
#         ax2.pie(
#             x = dist1['proportion'], 
#             labels = dist1['Ratio'],
#             autopct = lambda x: str(round(x, 2)) + '%',
#             pctdistance = 0.7, 
#             labeldistance = 1.1,
#             shadow = False,
#             textprops = {'fontsize': 8}
#         )
#         st.pyplot(fig2)

# with tab_b2:
#     col_b21, col_b22 = st.columns([1, 2], gap='small')
#     with col_b21:
#         st.dataframe(dist2)
#     with col_b22:
#         fig1, ax1 = plt.subplots()
#         sns.set_theme()
#         sns.barplot(x='Occurence', y='count', data=dist2)
#         st.pyplot(fig1)

# # Stats n°3
# st.markdown("##### Distribution de l'étendu des nuages par classe de nuage")

# g = sns.FacetGrid(dataframe_2, col='Label', height=4)
# g.map_dataframe(sns.histplot, 'CloudRatio', kde=True, bins=15)
# st.pyplot(plt)

# display_info("Commentaires ici...")

# # Stats n°4
# st.markdown("##### Quelques distributions...")

# exp1 = st.expander("Distribution du taux de couverture des nuages (par rapport à l'image entière) par classe de nuage", expanded=True)
# with exp1:
#     sns.set(style='whitegrid')
#     fig3, ax3 = plt.subplots(figsize=(8, 6))
#     sns.boxplot(x='Label', y='CloudRatio', data=dataframe_2, width=0.5)
#     sns.despine(left=True)
#     st.pyplot(fig3)

# exp2 = st.expander("Distribution de la moyenne des niveaux de pixel d'une zone nuageuse par classe de nuage")
# with exp2:
#     sns.set(style='whitegrid')
#     fig4, ax4 = plt.subplots(figsize=(8, 6))
#     sns.boxplot(x='Label', y='MeanPixelsCloud', data=dataframe_2, width=0.5)
#     sns.despine(left=True)
#     st.pyplot(fig4)

# exp3 = st.expander("Distribution de l'écart-type des niveaux de pixel d'une zone nuageuse par classe de nuage")
# with exp3:
#     sns.set(style='whitegrid')
#     fig5, ax5 = plt.subplots(figsize=(8, 6))
#     sns.boxplot(x='Label', y='StdPixelsCloud', data=dataframe_2, width=0.5)
#     sns.despine(left=True)
#     st.pyplot(fig5)

# ############################
# # Visualisation des images #
# ############################

# st.markdown("### 4. Visualisation des images")

# if st.checkbox("Afficher", key='xxx'):
#     with st.spinner("Veuillez patienter"):
#         exp1 = st.expander("Visualiser des images multi-classes")
#         with exp1:
#             ImageIds = ['002be4f_0', '002be4f_1', '002be4f_3']
#             showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, '../images/', hide_axis=True, show_mask=True)

#         exp2 = st.expander("Visualiser des images decrites par des segments compactes")
#         with exp2:
#             ImageIds = ['c22cfa7_3', 'f128b90_1', '6906aa0_3']
#             showImages(ImageIds, 1, 3, dataframe_2, 2100, 1400, '../images/', hide_axis=True, show_mask=True)

#         exp3 = st.expander("Visualiser des images mono-classe decrites par des segments disjoints")
#         with exp3:
#             ImageIds = ['db36366_2', '5717e63_3', '4a7b6e3_3', '37e8349_0', 'ae7a06d_3', 'c465c2e_3']
#             showImages(ImageIds, 2, 3, dataframe_2, 2100, 1400, '../images/', hide_axis=True, show_mask=True)

#         exp4 = st.expander("Visualiser des images présentant une zone cachée significative")
#         with exp4:
#             ImageIds = ['24884e7_0', 'f32724b_0', '3b9a092_0', '5265e81_0', '400a38d_0', 'a2dc5c0_0', '17fe76e_0', '838cd7a_0', 'fd5aa5d_0', '42ac1b7_0', '171e62f_0', '06e5dd6_0']
#             showImages(ImageIds, 3, 4, dataframe_2, 2100, 1400, '../images/', hide_axis=True, show_mask=False)