
The following is the directory structure -
code/
├── leaf_mask.zip         # Zipped dataset folder containing the dataset
├── coco_eval.py          # PyTorch utility functions for COCO evaluation
├── coco_utils.py         # PyTorch utility functions for handling COCO dataset
├── engine.py             # PyTorch utility functions for training and evaluation
├── mask_r_cnn.ipynb      # Main Jupyter notebook file to train and evaluate the model
├── transforms.py         # Utility functions for data augmentation and transformations
├── utils.py              # General utility functions used across the project

Steps to Run - 
1 - Extract leaf_mask.zip folder. Make sure the extracted folder has three sub folders - train, test, valid
2 - Run the mask_r_cnn.ipynb file