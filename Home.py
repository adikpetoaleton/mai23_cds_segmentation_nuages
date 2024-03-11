import streamlit as st

st.set_page_config(
    page_title='CloudIAtlas',
    layout='centered',
    page_icon=":one:"
)

#def main():

st.sidebar.title('CloudIAtlas Project')
# Chargez l'image
image = 'utils\cloudiatlas.png'

# Ajoutez l'image à la barre latérale
st.sidebar.image(image, caption='Référence littéraire et cinématographique', use_column_width=True)



st.title(":one: Segmentation de régions nuageuses")
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