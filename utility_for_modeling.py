import numpy as np
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def display_info(txt, color='Black'):
    st.write("<span style='color:" + color + ";'>_" + txt + "_</span>", unsafe_allow_html=True)

def display_info_list_items(items, color='Black'):
    st.write("- " + "\n- ".join(items))