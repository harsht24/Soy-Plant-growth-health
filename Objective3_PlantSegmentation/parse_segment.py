import os
import pandas as pd

# Define dataset directory
dataset_dir = "../Data"
output_csv = "Metadata.csv"

def extract_metadata(info_file):
    """Extract metadata from info.txt."""
    metadata = {}
    with open(info_file, 'r') as file:
        for line in file:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata

def parse_dataset(input_dir):
    """Parse dataset and extract metadata for each image."""
    data = []
    for main_folder in os.listdir(input_dir):
        main_path = os.path.join(input_dir, main_folder)
        if os.path.isdir(main_path):  # Ensure it's a directory
            for sub_folder in os.listdir(main_path):
                if sub_folder.startswith("Fluo_"):  # Only look for Fluo_ folders
                    fluo_path = os.path.join(main_path, sub_folder)
                    info_file = os.path.join(main_path, "info.txt")  # Info.txt is in the parent folder
                    
                    # Extract metadata from info.txt
                    if os.path.exists(info_file):
                        metadata = extract_metadata(info_file)
                    else:
                        metadata = {}

                    # Process images
                    for root, dirs, files in os.walk(fluo_path):
                        for file in files:
                            if file.endswith(".png"):  # Adjust for your image format
                                image_path = os.path.join(root, file)
                                data.append({
                                    "Image Path": image_path,
                                    "Folder": main_folder,
                                    "Subfolder": sub_folder,
                                    "Timestamp": metadata.get("Timestamp", ""),
                                    "IdTag": metadata.get("IdTag", ""),
                                    "Weight Before [g]": metadata.get("Weight before [g]", ""),
                                    "Weight After [g]": metadata.get("Weight after [g]", ""),
                                    "Water Amount [ml]": metadata.get("Water amount [ml]", ""),
                                    "Water Amount [g]": metadata.get("Water amount [g]", "")
                                })
    return data

# Parse dataset and create CSV
dataset_data = parse_dataset(dataset_dir)
df = pd.DataFrame(dataset_data)
df.to_csv(output_csv, index=False)

print(f"Metadata CSV saved at: {output_csv}")


