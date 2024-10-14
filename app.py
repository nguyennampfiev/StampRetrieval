# app.py
import streamlit as st
from stamp_extraction import process_image  # Adjust import as necessary
from util import load_json, find_similar_images
from mgpsa import build_retrieval_model, get_config
from retrieval_stamp import retrieval_stamp
from PIL import Image
import numpy as np
import os

query_feat = np.load("list_features_mpsa.npy").tolist()
query_name = np.load("list_name_features_mpsa.npy").tolist()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    print('Load retrieval model')
    config =get_config()
    model = build_retrieval_model(config,52)

    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Run Query"):
        # Call the processing function with the appropriate parameters
        with st.spinner("Extraction the biggest stamp in your image..."):  # Add a spinner while processing

            result = process_image(
                inputs=image_path,
                model='configs/config.py',
                weights='ckpt/ckpt.pth',
                texts='seal'
            )
            
            # Retrieve results using the retrieval_stamp function
            stamp_image = load_json()
            if stamp_image is None:
                st.error("Cannot find a stamp in the given documents.")
            else:
                st.image(stamp_image, caption='Extracted Stamp')
            feats = retrieval_stamp(model, stamp_image, device='cpu')
            list_results = find_similar_images(feats, query_feat, query_name)
            # Display results
            st.subheader("Retrieved Results:")
            
            # Assuming list_results contains paths to images or some data you want to show
            if isinstance(list_results, list) and len(list_results) > 0:
                for img_path in list_results:
                    if os.path.exists(img_path):
                        # Display each retrieved image
                        st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
                    else:
                        st.warning(f"Image path {img_path} does not exist.")
            else:
                st.info("No results found.")