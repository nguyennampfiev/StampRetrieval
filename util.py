import json
from PIL import Image
import torch
import numpy as np
import random
from scipy.spatial.distance import cdist
import os
import re
def clean_filename(filename):
    # Replace _<number> before the file extension
    #new_filename = re.sub(r'\s(_\d+|sceau[1-9]|seal)\s?', '', filename)
    #new_filename = re.sub(r'(_\d)', '', new_filename)
    new_filename = re.sub(r'\s(sceau[1-9])\s?', '', filename)
    new_filename = re.sub(r'(sceau[1-9])', '', new_filename)

    new_filename = re.sub(r'\s(seal)', '', new_filename)

    #pattern1 = r'^(N°\d+\s\w+)_\d+\.jpg$'
    pattern = r'^(N°\d+\s[\w\-]+)_\d+\.npy$'
    
    # Replace "_1" with nothing, keeping the rest of the name intact
    new_filename = re.sub(pattern, r'\1.npy', new_filename)    
    # Replace "_1" with nothing, keeping the rest of the name intact
    
    # Replace "_1" with nothing, keeping the rest of the name intact
    #new_filename = re.sub(pattern2, r'\1.jpg', new_filename)
    new_filename = re.sub(r'\.npy$', '.jpg', new_filename)
    return new_filename

def SetSeed(config):
    seed = config.misc.seed + config.local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def load_json(json_path='outputs/preds/uploaded_image.json', image_path='uploaded_image.jpg'):
    with open(json_path, 'r') as file:
        # Load the JSON data
        data = json.load(file)
        
    list_bboxes = data['bboxes']
    list_scores = data['scores']
    
    # Filter bounding boxes based on score threshold
    filtered_bboxes = [
        bbox for bbox, score in zip(list_bboxes, list_scores) if score > 0.5
    ]
    if len(filtered_bboxes)==0:
        return None
    # Find the bounding box with the largest area
    largest_area_bbox = None
    largest_area = 0

    for bbox in filtered_bboxes:
        # Assuming bbox is in the format [x1, y1, x2, y2]
        # Calculate area: (x2 - x1) * (y2 - y1)
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area > largest_area:
            largest_area = area
            largest_area_bbox = bbox

    original_image = Image.open(image_path).convert('RGB')

    # Crop the image to the largest bounding box coordinates
    if largest_area_bbox is not None:
        cropped_image = original_image.crop(largest_area_bbox)
        return cropped_image  # Return the cropped image
    else:
        return None  # No valid bounding box found
def load_json_v2(json_path='outputs/preds/uploaded_image.json', image_path='uploaded_image.jpg'):
    with open(json_path, 'r') as file:
        # Load the JSON data
        data = json.load(file)
        
    list_bboxes = data['bboxes']
    list_scores = data['scores']
    
    # Filter bounding boxes based on score threshold
    filtered_bboxes = [
        bbox for bbox, score in zip(list_bboxes, list_scores) if score > 0.5
    ]
    if len(filtered_bboxes)==0:
        return None
    # Find the bounding box with the largest area
    list_stamps = []
    original_image = Image.open(image_path).convert('RGB')
    for bbox in filtered_bboxes:
        # Assuming bbox is in the format [x1, y1, x2, y2]
        # Calculate area: (x2 - x1) * (y2 - y1)
        list_stamps.append(original_image.crop(bbox))
    return list_stamps
    
def find_similar_images(features, query_feat, query_name):
    # Implement your search logic here to find similar images
    features = features.to('cpu').detach().numpy()
    distances = cdist(features, query_feat, metric='cosine').squeeze()

    # Get the indices of results where the distance is less than 0.35
    top_results = np.where(distances < 0.35)[0]

    # Sort the selected indices based on the distances (ascending order)
    top_results_sorted = top_results[np.argsort(distances[top_results])]
    similar_images = []  # List of similar image paths or URLs
    # Placeholder logic for demonstration
    for i in range(len(top_results_sorted)):
        similar_images.append(os.path.join('./Full-Manuscript',clean_filename(query_name[top_results[i]])))
    return similar_images