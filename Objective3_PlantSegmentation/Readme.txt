# Plant Fluorescence Imaging Analysis

This repository contains scripts for analyzing plant health and growth using fluorescence imaging. The pipeline includes parsing dataset metadata, segmenting leaves, calculating fluorescence intensity metrics, and visualizing trends under various environmental conditions.

---

## Overview of Scripts

### 1. **Intensity_TV.py**
This script calculates fluorescence intensity metrics for cropped plant leaf regions. It extracts the red channel, masks leaf regions using a segmentation mask, and computes metrics such as total intensity, mean intensity, and normalized intensity.

#### Key Functions:
- **`crop_and_calculate_intensity_color`**:
  - Crops the red channel from fluorescent images based on a segmentation mask.
  - Calculates total intensity, mean intensity (excluding zero pixels), and total pixel count.
  - Outputs cropped images with red-channel visualization.
  
- **`process_and_crop_images_color`**:
  - Reads a metadata CSV file to process top-view fluorescence images (`Fluo_TV_90`).
  - Crops images using segmentation masks and saves metrics into a new CSV file.
  
- **`calculate_normalized_intensity`**:
  - Computes normalized intensity as total intensity divided by the number of leaf pixels.
  - Handles cases where no pixels are present (outputs zero).
  
- **`process_cropped_images_for_normalized_intensity`**:
  - Processes cropped images to calculate normalized intensity and saves the results into a CSV.

---

### 2. **parse_segment.py**
This script parses the dataset directory to extract metadata and generates a comprehensive CSV file with information about images and experimental details.

#### Key Functions:
- **`extract_metadata`**:
  - Reads `info.txt` files in dataset folders to extract metadata such as timestamps, plant weight, and water measurements.
  
- **`parse_dataset`**:
  - Recursively scans dataset directories for images under `Fluo_` subfolders.
  - Gathers image paths and metadata, saving the results in a CSV file for subsequent processing.

---

### 3. **Segment.py**
This script performs leaf segmentation on fluorescence images and calculates intensity metrics for each segmented leaf using advanced image processing techniques.

#### Key Functions:
- **`preprocess_image`**:
  - Extracts the red channel, normalizes pixel intensity, and applies Gaussian blur to denoise images.
  
- **`segment_leaves`**:
  - Segments leaf regions using Otsu's thresholding and morphological operations.
  - Removes small objects and labels connected components.
  
- **`detect_leaf_edges`**:
  - Detects leaf edges using Canny edge detection for visualization.
  
- **`calculate_leaf_intensities`**:
  - Computes intensity metrics for each segmented leaf, including total intensity and average intensity.
  
- **`save_visualizations`**:
  - Saves labeled images, leaf borders, and segmented masks for validation and analysis.
  
- **`process_and_segment_images`**:
  - Integrates preprocessing, segmentation, intensity calculations, and visualization steps for fluorescence images.

---

### 4. **Segment_TV.py**
This script processes top-view fluorescence images for leaf segmentation, focusing on generating intermediate outputs for debugging and validation.

#### Key Functions:
- **`preprocess_image`**:
  - Extracts and denoises the red channel from fluorescence images.
  
- **`segment_leaves`**:
  - Segments leaf regions and saves intermediate outputs, including thresholding results, morphological transformations, and small object removal.
  
- **`detect_leaf_edges`**:
  - Detects and saves leaf edges for visualization.
  
- **`save_visualizations`**:
  - Saves original images, segmentation masks, labeled images, and edge maps.
  
- **`process_and_segment_images`**:
  - Automates preprocessing, segmentation, edge detection, and visualization for all `Fluo_TV_90` images.

---

### 5. **visualizeplot.py**
This script analyzes processed results and organizes them into condition-specific datasets for visualization and trend analysis.

#### Key Features:
- **Data Loading**:
  - Reads metadata and normalized intensity results into Pandas DataFrames.
  
- **Condition Filtering**:
  - Filters data based on conditions (e.g., `LTN_WW`, `LTN_D`, `HTN_WW`, `HTN_D`) to separate datasets for specific environmental scenarios.
  
- **Merging and Exporting**:
  - Merges metadata with intensity results and saves condition-specific datasets into separate CSV files.
  
- **Validation**:
  - Verifies the consistency of total images and sub-dataset splits to ensure correctness.

---

## Workflow and Execution Steps

### **1. Parse the Dataset**
Run `parse_segment.py` to generate the metadata CSV file with image paths and experimental details.

**Example:**
```bash
python parse_segment.py

### 2. Segment Fluorescence Images
Use `Segment.py` or `Segment_TV.py` to preprocess, segment, and visualize the fluorescence images.

**Example:**
```bash
python Segment.py

python Intensity_TV.py


python Intensity_TV.py

python visualizeplot.py
```

** Requirements **
- Python 3.8 or higher

** Required Libraries: **
- cv2 (OpenCV)
- numpy
- pandas
- matplotlib
- skimage

