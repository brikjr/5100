import streamlit as st
import pandas as pd
import numpy as np
import model_image_analyzer as OM

st.title('Image processor')

uploaded_file = st.file_uploader("Please upload an image of the bookshelf.", ['jpg', 'jpeg', 'heic', 'png'], )
if(uploaded_file != None):
    st.image(uploaded_file, caption="Bookshelf Image")

if st.button("Process Bookshelf", type="primary"):
    with st.spinner('Processing Image, Please Wait'):
        book_list = OM.processImageGetList(uploaded_file)
    st.write("Books in image:")
    for title in book_list:
        st.markdown("- " + title)

