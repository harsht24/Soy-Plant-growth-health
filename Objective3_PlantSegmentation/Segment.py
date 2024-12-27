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
dataset_dir = "../SOY_fluorescent_images"
output_csv = "Metadata.csv"


def preprocess_image(image_path):
    """Preprocess the image: extract red channel and denoise."""
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]  # Extract red channel
    red_channel = cv2.normalize(red_channel, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    red_channel = cv2.GaussianBlur(red_channel, (5, 5), 0)  # Denoise
    return red_channel

def segment_leaves(image):
    """Segment the leaves using Otsu's thresholding and morphological operations."""
    thresh_val = threshold_otsu(image)
    binary_mask = image > thresh_val
    # Morphological operations
    kernel = disk(5)
    binary_mask = closing(opening(binary_mask, kernel), kernel)
    # Remove small objects
    binary_mask = remove_small_objects(binary_mask, min_size=500)
    # Label connected components
    labeled_image = label(binary_mask)
    return labeled_image, binary_mask

def detect_leaf_edges(binary_mask):
    """Detect edges (borders) of the leaves using Canny edge detection."""
    edges = canny(binary_mask, sigma=2)
    return edges

def calculate_leaf_intensities(labeled_image, original_image):
    """Calculate intensity metrics for each leaf."""
    props = regionprops(labeled_image, intensity_image=original_image)
    leaf_metrics = []
    for prop in props:
        leaf_id = prop.label
        area = prop.area
        total_intensity = prop.mean_intensity * prop.area
        avg_intensity = prop.mean_intensity
        leaf_metrics.append({
            "Leaf ID": leaf_id,
            "Area (pixels)": area,
            "Total Intensity": total_intensity,
            "Average Intensity": avg_intensity
        })
    return leaf_metrics

def save_visualizations(image_folder, image_name, labeled_image, edges):
    """Save labeled leaves and edge visualizations as images."""
    labeled_image_path = os.path.join(image_folder, f"{image_name}_labeled_leaves.png")
    edges_image_path = os.path.join(image_folder, f"{image_name}_leaf_edges.png")
    # Save labeled image
    plt.figure(figsize=(8, 8))
    plt.imshow(labeled_image, cmap='jet')
    plt.title('Labeled Leaves')
    plt.axis('off')
    plt.savefig(labeled_image_path, bbox_inches='tight')
    plt.close()
    # Save edges image
    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')
    plt.title('Leaf Borders (Edges)')
    plt.axis('off')
    plt.savefig(edges_image_path, bbox_inches='tight')
    plt.close()
    print(f"Labeled Leaves Saved: {labeled_image_path}")
    print(f"Leaf Borders Saved: {edges_image_path}")


def process_and_segment_images(csv_file):
    """Process images, refine segmentation, detect edges, and save results in the same folder."""
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        image_path = row["Image Path"]
        image_folder = os.path.dirname(image_path)  # Get the folder of the image
        image_name = os.path.basename(image_path).split(".")[0]
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        # Segment leaves
        labeled_image, binary_mask = segment_leaves(preprocessed_image)
        # Detect edges (borders) of leaves
        leaf_edges = detect_leaf_edges(binary_mask)
        # Save refined segmented image in the same folder
        segmented_image_path = os.path.join(image_folder, f"{image_name}_segmented.png")
        cv2.imwrite(segmented_image_path, (binary_mask * 255).astype(np.uint8))
        # Save visualizations for labeled leaves and leaf borders
        save_visualizations(image_folder, image_name, labeled_image, leaf_edges)
        # Calculate intensities for refined segmentation
        leaf_metrics = calculate_leaf_intensities(labeled_image, preprocessed_image)
        # Save leaf intensity metrics as CSV in the same folder
        leaf_metrics_csv_path = os.path.join(image_folder, f"{image_name}_leaf_metrics.csv")
        leaf_df = pd.DataFrame(leaf_metrics)
        leaf_df.to_csv(leaf_metrics_csv_path, index=False)
        print(f"Processed: {image_path}")
        print(f"Segmented Image Saved: {segmented_image_path}")
        print(f"Leaf Metrics Saved: {leaf_metrics_csv_path}")


# Run the pipeline
process_and_segment_images(output_csv)
