import streamlit as st
import random
import os

from utils import *
from utils_unet import load_model_stripe, load_model_unet, predict_and_display

st.set_page_config(page_title="Cloud-Net", page_icon=":five:")


@st.cache_resource()
def load_model_unet_cached():
    return load_model_unet()

@st.cache_resource()
def load_model_stripe_cached():
    return load_model_stripe()

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

model_unet = load_model_unet_cached()
model_stripe = load_model_stripe_cached()

st.title(":three: Modélisation avec U-Net")
display_info("Espace de test de l'architecture U-Net pour la segmentation sémantique des régions nuageuses")

st.write("<br><p style='text-align: justify;'>Le parcours décrit ci-dessous aura pour objectif de \
         <br>Optimiser l'apprentissage d'un modèle de segmentation et classification \
         <br>Sous contrainte d'un contexte technique limité (RAM, ...) \
         <br>Et de données initiales de qualité moyennes", unsafe_allow_html=True)

st.header("Présentation U-Net & Paradoxe de la mesure")
st.write("<br><p style='text-align: justify;'>L’architecture U-Net a été développée pour la segmentation d’images biomédicales. Elle est basée sur l’architecture entièrement convolutionnelle, modifiée pour fonctionner avec moins d’images d’entraînement et permettre une segmentation plus précise.\
<br><br>Dans le cadre d’un projet ML d’identification de formes de nuages sur des photos satellites, donc de vision par ordinateur, l’usage de U-Net se justifie par sa capacité à effectuer une segmentation sémantique (segmentation + classification), c’est-à-dire à classer chaque pixel d’une image comme appartenant à une classe particulière.", unsafe_allow_html=True)

# unet_image = 'utils\unet.png'
# unet_image = 'utils\\unet.png'
st.image('utils\\unet.png', caption='U-Net architecture', use_column_width=True)

st.write("<br><p style='text-align: justify;'>Alors c'est quoi ce paradoxe de la mesure ? \
         <br>Et pourquoi ce titre pseudo énigmatique !? Pour ne pas que tu t'endormes mon enfant !!!", unsafe_allow_html=True)
st.image('utils\\paradoxe.png', use_column_width=True)
st.write("<br><p style='text-align: justify;'>D'abord, que mesure t'on justement ?", unsafe_allow_html=True)
st.image('utils\\iou.png')

st.header(":sun_behind_cloud: Un mot sur les dataset")

st.write("<br><p style='text-align: justify;'>L'alimentation des modèles s'est appuyé sur plusieurs datasets constitués de : \
         <br>1. L'image modifiée en noir et blanc et au format 256\*256 \
         <br>2. Un masque par classe analysée au format 256\*256 indiquant les pixels d'intérêts par un 1, les autres par un 0", unsafe_allow_html=True)

st.header(":+1: Ce qui a été fait")
st.subheader("Pour commencer : les boulettes :shit: !")

st.write("<br><p style='text-align: justify;'>:white_check_mark: L'IA, c'est comme un chat, c'est fainéant. Donc le background comme masque supplémentaire à 54% ... \
         <br>Bah l'IA elle est contente à 54% en considérant que tout est background", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:white_check_mark: Le taux d'apprentissage mal établi qui te fait exploser le modèle, ça marche pas non plus", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:white_check_mark: Et pleins d'autres, mais ça prendrait tout le temps disponible", unsafe_allow_html=True)

st.subheader("Pour suivre : les autres approches :mechanical_arm: !")
st.write("<br><p style='text-align: justify;'>D'abord, les optimisations techniques liées à la taille du modèle (Et au coût de la formation \
         empêchant de se payer du calcul sur Google :smile:)", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:white_check_mark: Utilisation des formats uint8 et bfloat16", unsafe_allow_html=True)

st.write("<br><p style='text-align: justify;'>Ensuite les différentes approches de dataset fournis au modèle", unsafe_allow_html=True)

col10, col20, col30 = st.columns(3)

col10.write("<br>Modèle standard sans modification du dataset", unsafe_allow_html=True)
col20.write("<br>Un résultat qui restera l'étalon pour la suite des essais", unsafe_allow_html=True)
col30.image('utils\\etalon.png')

col11, col21, col31 = st.columns(3)

col11.write("<br>Background en tant que masque", unsafe_allow_html=True)
col21.write("<br>L'IA optimise son pourcentage en imputant tout au background", unsafe_allow_html=True)

col12, col22, col32 = st.columns(3)

col12.write("<br>Avec augmentation des images", unsafe_allow_html=True)
col22.write("<br>Le choix de la méthode d'augmentation à revoir car augmentation par images de classes uniques pour équilibrage des classes", unsafe_allow_html=True)
col32.image('utils\\background.png')

col13, col23, col33 = st.columns(3)

col13.write("<br>Avec filtre sur le niveau de gris", unsafe_allow_html=True)
col23.write("<br>Résultats de pixels d'intérêts trop répartis pour un apprentissage correct", unsafe_allow_html=True)
col33.image('utils\\filter.png')

col14, col24, col34 = st.columns(3)

col14.write("<br>Avec filtre sur le niveau de gris et morphisme", unsafe_allow_html=True)
col24.write("<br>Résultats bons mais inférieurs à l'étalon obtenu", unsafe_allow_html=True)
col34.image('utils\\filter_morph.png')

col15, col25, col35 = st.columns(3)

col15.write("<br>Avec masque indiquant la bande noire", unsafe_allow_html=True)
col25.write("<br>Résultats bons mais insuffisants par rapport à l'attente", unsafe_allow_html=True)
col35.image('utils\\unet_stripemask.png')

col15, col25, col35 = st.columns(3)

col15.write("<br>Modèle U-Net light indépendant pour la bande noire", unsafe_allow_html=True)
col25.write("<br>Résultats très bons à combiner au modèle étalon", unsafe_allow_html=True)
col35.image('utils\\stripe_model.png')

st.header(":-1: Ce qui n'a pas été fait (mais que ça aurait été cool :zany_face:)")

st.write("<br><p style='text-align: justify;'>:x: Evolution du poids de Sugar sous-représenté dans les sorties", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:x: 1 neurone d'output du modèle par classe + stripe afin d'unifier et gérer les recouvrements", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:x: Cross-validation avec RAM suffisante", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:x: Format PNG ou TIFF des images pour éviter la perte de compression du jpeg", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:x: Augmentation du dataset sur un autre critère d'équilibrage", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:x: Transfer learning depuis modèle entraîné aux images fractales pour apprentissage de la symétrie évolutive", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>:x: Autres modèles : Double U-Net et autres Net (UAT, RIU), transposed convolution", unsafe_allow_html=True)

st.header("Et concrètement, ça donne quoi ?")

st.write("<br><p style='text-align: justify;'>Le modèle final qui vous est proposé ci-dessous est l'application du modèle étalon surchargé du modèle des bandes noires", unsafe_allow_html=True)
st.write("<br><p style='text-align: justify;'>Cette combinaison avec un MeanIoU de 54% et quasi 100% de bande noire conduit à un meanIoU combiné de 54 / (100 - 10) soit :", unsafe_allow_html=True)
st.markdown("**60%**")

st.sidebar.divider()

st.write("<br><p style='text-align: justify;'>Chose promise, chose due, on teste !!!", unsafe_allow_html=True)

image_folder = os.path.join(os.path.dirname(os.getcwd()), 'mai23_cds_segmentation_nuages\images')
image_list = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

if not image_list:
    st.write("Le dossier 'images' ne contient pas d'images .jpg. ET c'est pas bon ça !!!")
else:

    slide_col, text_col, random_col = st.columns([10,3,5])

    # Créer un slider pour choisir l'image
    slider = slide_col.slider('Choisissez une image', 1, len(image_list), 1)
    st.session_state.selected_image = image_list[slider - 1]
    text_col.text('ou')

    # Ajouter un bouton pour choisir une image au hasard
    if random_col.button('Choisir au hasard'):
        st.session_state.selected_image = random.choice(image_list)

    # Afficher l'image sélectionnée et son nom
    st.image(os.path.join(image_folder, st.session_state.selected_image))
    predict_and_display(st.session_state.selected_image, model_unet, model_stripe)

