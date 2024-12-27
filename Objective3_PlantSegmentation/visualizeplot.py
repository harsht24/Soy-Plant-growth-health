# Re-import libraries and reload the datasets after reset
import pandas as pd

# Reload necessary files
metadata_file = 'Metadata.csv'
normalized_intensity_file = 'NormalizedIntensityResults.csv'

# Load datasets
metadata_df = pd.read_csv(metadata_file)
normalized_intensity_df = pd.read_csv(normalized_intensity_file)

# Filter the metadata for "Top View" data
top_view_metadata = metadata_df[metadata_df['Subfolder'].str.contains('Fluo_TV_90', na=False)]

# Merge the top view metadata with the normalized intensity data
merged_top_view = pd.merge(top_view_metadata, normalized_intensity_df, on="Image Path", how="inner")

# Define conditions for the four sub-datasets
conditions = {
    "LTN_WW": (merged_top_view['Folder'].str.contains('LTN', na=False)) & (merged_top_view['IdTag'].str.contains('WW', na=False)),
    "LTN_D": (merged_top_view['Folder'].str.contains('LTN', na=False)) & (merged_top_view['IdTag'].str.contains('D', na=False)),
    "HTN_WW": (merged_top_view['Folder'].str.contains('HTN', na=False)) & (merged_top_view['IdTag'].str.contains('WW', na=False)),
    "HTN_D": (merged_top_view['Folder'].str.contains('HTN', na=False)) & (merged_top_view['IdTag'].str.contains('D', na=False))
}

# Create sub-datasets based on the conditions
sub_datasets = {name: merged_top_view[condition] for name, condition in conditions.items()}

# Save the sub-datasets to CSV files
output_files = {}
for name, dataset in sub_datasets.items():
    output_file = f"{name}_Dataset.csv"
    dataset.to_csv(output_file, index=False)
    output_files[name] = output_file

    # Verify the total number of top view images in the merged dataset
total_top_view_images = merged_top_view.shape[0]

# Verify the sum of images in the sub-datasets
sub_datasets_total_images = sum([dataset.shape[0] for dataset in sub_datasets.values()])

total_top_view_images, sub_datasets_total_images  # Display total and summed counts

print(sub_datasets_total_images,total_top_view_images)

output_files  # Display the paths to the saved datasets
