# app.py
from stamp_extraction import process_image  # Adjust import as necessary
from util import load_json_v2, find_similar_images
from mgpsa import build_retrieval_model, get_config
from retrieval_stamp import retrieval_stamp
from PIL import Image
import numpy as np
import os
import gradio as gr
import tempfile 
from typing import List, Union, Tuple



print('Loading retrieval model')
config =get_config()
model = build_retrieval_model(config,52)
print('Loading saved features')
query_feat = np.load("list_features_mpsa.npy").tolist()
query_name = np.load("list_name_features_mpsa.npy").tolist()



def convert_to_pil_image(img) -> Image.Image:
    """Convert various image formats to PIL Image"""
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img)
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

def extract_stamps(input_img: Union[np.ndarray, None]) -> Tuple[list, str]:
    """First step: Extract stamps from the input image"""
    if input_img is None or (isinstance(input_img, np.ndarray) and input_img.size == 0):
        raise gr.Error("No input image provided. Please upload an image.")
        
    try:
        # Convert NumPy array to PIL Image
        input_pil_image = Image.fromarray(input_img)
        
        image_path='uploaded_image.jpg'
        input_pil_image.save(image_path)
            
        result = process_image(
            inputs=image_path,
            model='configs/config.py',
            weights='ckpt/ckpt.pth',
            texts='seal'
        )
        
        # Retrieve results using the retrieval_stamp function
        stamp_images = load_json_v2()  # Assuming this returns a list of images
        if not stamp_images:
            raise gr.Error("No stamps found in the uploaded document. Please try with a different image.")
            
        # Convert stamps to format compatible with Gradio
        processed_stamps = []
        for stamp in stamp_images:
            # Convert each stamp to PIL Image if it's not already
            pil_stamp = convert_to_pil_image(stamp)
            # Save to temporary file and store the path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_stamp.save(temp_file.name)
                processed_stamps.append(temp_file.name)
            
        return processed_stamps, f"✓ Found {len(processed_stamps)} stamps. Please select one for retrieval."
        
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"An error occurred while extracting stamps: {str(e)}")

def retrieve_similar_documents(selected_index: gr.SelectData) -> Tuple[list, str]:
    """Second step: Process the selected stamp and find similar documents"""
    try:
        dict_of_selected_stamp_path = selected_index.value
        selected_stamp_path = dict_of_selected_stamp_path['image']['path']
        if not selected_stamp_path or not os.path.exists(selected_stamp_path):
            raise gr.Error("Please select a valid stamp first.")
        
        # Load the selected stamp image
        stamp_image = Image.open(selected_stamp_path)
        
        print('Processing selected stamp for retrieval...')
        feats = retrieval_stamp(model, stamp_image, device='cpu')
        list_results = find_similar_images(feats, query_feat, query_name)
        
        if not list_results:
            raise gr.Error("No matching documents found for the selected stamp.")
            
        # Get document names from paths
        document_names = [os.path.basename(img_path).replace('.png', '').replace('_', ' ').title() 
                         for img_path in list_results]
        
        return list_results, f"✓ Found {len(list_results)} matching documents: {', '.join(document_names)}"
        
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"An error occurred while retrieving similar documents: {str(e)}")

# Create the Gradio interface with two-step process
with gr.Blocks(title="Document Retrieval System", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Document Retrieval System")
    gr.Markdown("Upload a document to find similar documents based on stamp matching")
    
    with gr.Row():
        # Input section
        with gr.Column():
            input_image = gr.Image(label="Upload Document")
            extract_btn = gr.Button("Extract Stamps")
        
        # Stamps selection section
        with gr.Column():
            stamps_gallery = gr.Gallery(
                label="Extracted Stamps (Click to Select)",
                show_label=True,
                visible=True,
                height="auto"
            )
            stamps_status = gr.Textbox(label="Extraction Status", visible=True)
            # retrieve_btn = gr.Button("Find Similar Documents", visible=True)
    
    # Results section
    with gr.Column():
        similar_docs = gr.Gallery(
            label="Similar Documents Found",
            show_label=True,
            visible=True,
            height="auto"
        )
        retrieval_status = gr.Textbox(label="Retrieval Status", visible=True)
    
    # Event handlers
    extract_btn.click(
        fn=extract_stamps,
        inputs=[input_image],
        outputs=[stamps_gallery, stamps_status]
    )
    
    # Use select event instead of click button
    stamps_gallery.select(
        fn=retrieve_similar_documents,
        inputs=None,  # The selection data is passed automatically
        outputs=[similar_docs, retrieval_status]
    )
    

# Launch the interface
demo.launch(show_error=True, quiet=False)
