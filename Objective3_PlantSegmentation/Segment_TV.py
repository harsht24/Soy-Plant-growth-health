import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.feature import canny
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Paths
output_csv = "Metadata.csv"
output_base_dir = "topview_segmented_Data"  # New output directory

def preprocess_image(image_path):
    """Preprocess the image: extract red channel and denoise."""
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]  # Extract red channel
    red_channel = cv2.normalize(red_channel, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    red_channel = cv2.GaussianBlur(red_channel, (5, 5), 0)  # Denoise
    return red_channel

def segment_leaves(image, output_folder, image_name):
    """Segment the leaves using Otsu's thresholding and morphological operations."""
    # Thresholding
    thresh_val = threshold_otsu(image)
    binary_mask = image > thresh_val
    thresholding_path = os.path.join(output_folder, f"{image_name}_thresholding.png")
    cv2.imwrite(thresholding_path, (binary_mask * 255).astype(np.uint8))

    # Morphological operations
    kernel = disk(5)
    binary_mask = closing(opening(binary_mask, kernel), kernel)
    morph_path = os.path.join(output_folder, f"{image_name}_morphology.png")
    cv2.imwrite(morph_path, (binary_mask * 255).astype(np.uint8))

    # Remove small objects
    binary_mask = remove_small_objects(binary_mask, min_size=500)
    small_object_removal_path = os.path.join(output_folder, f"{image_name}_small_object_removal.png")
    cv2.imwrite(small_object_removal_path, (binary_mask * 255).astype(np.uint8))

    # Label connected components
    labeled_image = label(binary_mask)
    return labeled_image, binary_mask

def detect_leaf_edges(binary_mask, output_folder, image_name):
    """Detect edges (borders) of the leaves using Canny edge detection."""
    edges = canny(binary_mask, sigma=2)
    edges_path = os.path.join(output_folder, f"{image_name}_edges.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')
    plt.title('Leaf Borders (Edges)')
    plt.axis('off')
    plt.savefig(edges_path, bbox_inches='tight')
    plt.close()
    return edges

def save_visualizations(output_folder, image_name, original_image, labeled_image, edges, binary_mask):
    """Save original, segmented, and visualized images in the structured folder."""
    # Create folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Save original image
    original_image_path = os.path.join(output_folder, f"{image_name}_original.png")
    cv2.imwrite(original_image_path, original_image)

    # Save segmented mask
    segmented_image_path = os.path.join(output_folder, f"{image_name}_segmented.png")
    cv2.imwrite(segmented_image_path, (binary_mask * 255).astype(np.uint8))

    # Save labeled image visualization
    labeled_image_path = os.path.join(output_folder, f"{image_name}_labeled_leaves.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(labeled_image, cmap='jet')
    plt.title('Labeled Leaves')
    plt.axis('off')
    plt.savefig(labeled_image_path, bbox_inches='tight')
    plt.close()

    print(f"Saved: {original_image_path}, {segmented_image_path}, {labeled_image_path}")

def process_and_segment_images(csv_file, output_dir):
    """Process only Fluo_TV_90 images, save in the desired structured folder."""
    df = pd.read_csv(csv_file)
    # Filter for Fluo_TV_90 subfolder
    filtered_df = df[df["Subfolder"] == "Fluo_TV_90"]

    for index, row in filtered_df.iterrows():
        image_path = row["Image Path"]
        experiment_folder = row["Folder"]
        image_name = os.path.basename(image_path).split(".")[0]

        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)

        # Create output folder structure
        experiment_output_folder = os.path.join(output_dir, experiment_folder)
        os.makedirs(experiment_output_folder, exist_ok=True)

        # Save preprocessed image
        preprocessed_path = os.path.join(experiment_output_folder, f"{image_name}_preprocessed.png")
        cv2.imwrite(preprocessed_path, preprocessed_image)

        # Segment leaves
        labeled_image, binary_mask = segment_leaves(preprocessed_image, experiment_output_folder, image_name)

        # Detect edges
        leaf_edges = detect_leaf_edges(binary_mask, experiment_output_folder, image_name)

        # Save visualizations and processed images
        save_visualizations(
            experiment_output_folder,
            image_name,
            cv2.imread(image_path),
            labeled_image,
            leaf_edges,
            binary_mask
        )

# Run the updated pipeline
process_and_segment_images(output_csv, output_base_dir)
