import streamlit as st

st.set_page_config(
    page_title='CloudIAtlas',
    layout='centered',
    page_icon=":one:"
)

# #def main():

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

# Debug

@st.cache_data
def load_Initial_data():
    data = pd.read_csv('clouds_0.csv')
    return data

@st.cache_data
def load_cleaned_data():
    data = pd.read_csv('clouds_1.csv')
    return data

@st.cache_data
def load_enriched_data():
    data = pd.read_csv('clouds_2.csv')
    return data

@st.cache_data
def load_hist_data():
    data = pd.read_csv('training_history.csv')
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

if 'dataframe_4' not in st.session_state:
    st.session_state.dataframe_4 = None
    dataframe_4 = None
else:
    dataframe_4 = st.session_state.dataframe_4

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

# Debug


st.title(":one: ClouIAtlas : Segmentation de régions nuageuses")
st.write("<br><p style='text-align: justify;'>Depuis de nombreuses années, le changement climatique est au cœur des préoccupations\
des scientifiques et au premier plan des décisions politiques importantes. Les scientifiques,\
comme ceux de l'Institut Max Planck de météorologie, mènent de nouvelles recherches sur\
l'atmosphère mondiale en constante évolution.\
<br><br>Les nuages peu profonds jouent un rôle essentiel dans la détermination du climat de la\
Terre. Ils sont également difficiles à comprendre et à représenter dans les modèles\
climatiques. En classant les différents types d'organisation des nuages (Poisson, Fleure, \
Gravier, Sucre), les chercheurs de Max Planck espèrent améliorer la compréhension\
physique de ces nuages, ce qui aidera à construire de meilleurs modèles climatiques.\
<br><br>Les nuages peuvent s'organiser de nombreuses façons, mais les limites entre les différentes\
formes d'organisation sont floues. Il est donc difficile d'élaborer des algorithmes traditionnels\
basés sur des règles pour séparer les caractéristiques des nuages. L'œil humain,\
cependant, est très doué pour détecter les caractéristiques, telles que les nuages qui\
ressemblent à des fleurs.\
<br><br>L’objectif de ce projet est d’analyser et d’identifier dans chaque image satellite les régions\
qui contiennent une formation nuageuse particulière. Il s’agira ensuite d’élaborer un modèle\
permettant de classer les modèles d'organisation des nuages à partir de ces images\
satellite.\
<br><br>Cette recherche guidera le développement de modèles de nouvelle génération qui\
pourraient réduire les incertitudes des projections climatiques</p><br>", unsafe_allow_html=True)

st.image("Teaser_AnimationwLabels.gif")

# if __name__ == '__main__':
#     main()