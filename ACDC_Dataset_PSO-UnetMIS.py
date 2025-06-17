import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K

# Kiểm tra xem TensorFlow có nhận diện được GPU không
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/GPU:0'):  # Sử dụng GPU đầu tiên
    import os
    import glob
    import numpy as np
    import nibabel as nib  # Import nibabel to handle NIfTI (.nii) files
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import cv2

    # Path to the ACDC dataset
    data_path = 'C:/Users/user2/Desktop/Phu1/ACDC_dataset/database/'

    # Folders for training and testing data
    train_path = os.path.join(data_path, 'training')
    test_path = os.path.join(data_path, 'testing')

    # Initialize lists to store image and mask paths
    train_images = []
    train_masks = []
    test_images = []
    test_masks = []

    # Load training images and masks
    for i in range(1, 101):  # 100 patients in total
        patient_path = os.path.join(train_path, f'patient{i:03d}')  # Correct patient path formatting
        
        # Debugging: Print the patient path
        print(f"Loading data from: {patient_path}")
        
        images = sorted(glob.glob(os.path.join(patient_path, f'patient{i:03d}_frame01.nii')))
        masks = sorted(glob.glob(os.path.join(patient_path, f'patient{i:03d}_frame01_gt.nii')))
        
        # Debugging: Print the number of files found
        print(f"Found {len(images)} images and {len(masks)} masks for patient {i}")
        
        # Check if the directories exist
        if len(images) == 0 or len(masks) == 0:
            print(f"Warning: No images or masks found for patient {i}")
        
        train_images.extend(images)
        train_masks.extend(masks)

    # Load testing images and masks
    for i in range(101, 151):  # 50 patients for testing
        patient_path = os.path.join(test_path, f'patient{i:03d}')  # Correct patient path formatting
        
        # Debugging: Print the patient path
        print(f"Loading data from: {patient_path}")
        
        images = sorted(glob.glob(os.path.join(patient_path, f'patient{i:03d}_frame01.nii')))
        masks = sorted(glob.glob(os.path.join(patient_path, f'patient{i:03d}_frame01_gt.nii')))
        
        # Debugging: Print the number of files found
        print(f"Found {len(images)} images and {len(masks)} masks for patient {i}")
        
        # Check if the directories exist
        if len(images) == 0 or len(masks) == 0:
            print(f"Warning: No images or masks found for patient {i}")
        
        test_images.extend(images)
        test_masks.extend(masks)

    # Combine images and masks from both training and testing sets
    image_paths = train_images + test_images
    mask_paths = train_masks + test_masks

    # Initialize lists for processed images and masks
    Images = []
    Masks = []
    img_size = 64  # Resize images to 64x64 for memory efficiency

    # Process images using nibabel
    for img_path in image_paths:
        print(f"Processing image: {img_path}")  # Debugging
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        img_nii = nib.load(img_path)  # Load the .nii file using nibabel
        img = img_nii.get_fdata()  # Get the image data as a numpy array
        print(f"Image shape before resize: {img.shape}")

        # Check if the image is 3D, and take the first slice if so
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take the first slice if it's 3D

        # Now resize the 2D image
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float32') / 255.0
        Images.append(img_resized[..., np.newaxis])

    # Process masks using nibabel
    for mask_path in mask_paths:
        print(f"Processing mask: {mask_path}")  # Debugging
        if not os.path.exists(mask_path):
            print(f"File not found: {mask_path}")
            continue

        mask_nii = nib.load(mask_path)  # Load the .nii file using nibabel
        mask = mask_nii.get_fdata()  # Get the mask data as a numpy array
        print(f"Mask shape before resize: {mask.shape}")

        # Check if the mask is 3D, and take the first slice if so
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take the first slice if it's 3D

        # Now resize the 2D mask
        mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float32') / 255.0
        Masks.append(mask_resized[..., np.newaxis])

    # Convert lists to numpy arrays
    Images = np.array(Images)
    Masks = np.array(Masks)
    
    # Check if data has been loaded correctly
    print(f"Loaded {Images.shape[0]} images and {Masks.shape[0]} masks.")
    if Images.shape[0] == 0 or Masks.shape[0] == 0:
        print("Error: No images or masks loaded. Please check the file paths.")
        exit()

    # Total number of patients
    train_patient = np.arange(1, 101)
    test_patient = np.arange(101, 151)

    # First, split into 70% for training and 30% for temporary validation+test
    train_patient_ids, val_patient_ids = train_test_split(train_patient, train_size=0.7, test_size=0.1, random_state=42)

    # Now, split the 30% temporary set into 15 patients for validation and 30 for testing
    test_patient_ids, _ = train_test_split(test_patient, test_size=0.39, random_state=42)  # 10% for validation out of the remaining 30%

    # Check the splits
    print("Train Patient IDs:", train_patient_ids)
    print("Validation Patient IDs:", val_patient_ids)
    print("Test Patient IDs:", test_patient_ids)

    # Optionally, print the number of patients in each set
    print(f"Training set size: {len(train_patient_ids)}")
    print(f"Validation set size: {len(val_patient_ids)}")
    print(f"Test set size: {len(test_patient_ids)}")

    
    # Initialize lists for training, validation, and test images and masks
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    # Define the image size (e.g., 256x256)
    img_size = 256
    target_depth = 256  # Desired number of slices

    def pad_volume(volume, target_depth):
        """Pad or trim the volume to match the target depth."""
        depth = volume.shape[0]
        if depth < target_depth:
            # Pad the volume with zeros (black slices) to the target depth
            padding = ((0, target_depth - depth), (0, 0), (0, 0))
            volume = np.pad(volume, padding, mode='constant', constant_values=0)
        elif depth > target_depth:
            # Trim the volume if it exceeds the target depth
            volume = volume[:target_depth]
        return volume

    def load_data(patient_ids, data_path, set_type='train'):
        X_data = []
        y_data = []
        
        for i in patient_ids:
            patient_path = os.path.join(data_path, f'patient{i:03d}')
            
            images = sorted(glob.glob(os.path.join(patient_path, f'patient{i:03d}_frame01.nii')))
            masks = sorted(glob.glob(os.path.join(patient_path, f'patient{i:03d}_frame01_gt.nii')))
            
            for img_path, mask_path in zip(images, masks):
                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    print(f"File not found: {img_path} or {mask_path}")
                    continue
                
                img_nii = nib.load(img_path)
                mask_nii = nib.load(mask_path)
                
                img_data = img_nii.get_fdata()
                mask_data = mask_nii.get_fdata()
                
                # Check if the data is 3D
                if img_data.ndim == 3:
                    # Pad or trim the volumes to match the target depth
                    img_data = pad_volume(img_data, target_depth)
                    mask_data = pad_volume(mask_data, target_depth)
                    
                    # Resize each slice (frame) individually
                    img_resized = np.zeros((img_data.shape[0], img_size, img_size), dtype='float32')
                    mask_resized = np.zeros((mask_data.shape[0], img_size, img_size), dtype='float32')
                    
                    for idx in range(img_data.shape[0]):
                        img_resized[idx] = cv2.resize(img_data[idx], (img_size, img_size), interpolation=cv2.INTER_AREA)
                        mask_resized[idx] = cv2.resize(mask_data[idx], (img_size, img_size), interpolation=cv2.INTER_AREA)
                    
                    # Normalize the image and mask (optional)
                    img_resized = img_resized / 255.0
                    mask_resized = mask_resized / 255.0
                    
                    # Append to the data lists
                    X_data.append(img_resized[..., np.newaxis])
                    y_data.append(mask_resized[..., np.newaxis])
                    
                else:
                    print(f"Unexpected image shape: {img_data.shape}, skipping patient {i}")
        
        # Ensure that X_data and y_data have the same shape and are homogeneous
        try:
            X_data = np.array(X_data)
            y_data = np.array(y_data)
        except ValueError as e:
            print(f"Error converting to numpy array: {e}")
            print(f"X_data shape: {len(X_data)} items with shapes {[x.shape for x in X_data]}")
            print(f"y_data shape: {len(y_data)} items with shapes {[y.shape for y in y_data]}")
            raise
        
        return X_data, y_data

    # Load data for training, validation, and testing
    X_train, y_train = load_data(train_patient_ids, train_path, 'train')
    X_val, y_val = load_data(val_patient_ids, train_path, 'val')
    X_test, y_test = load_data(test_patient_ids, test_path, 'test')
    # Print dataset sizes
    print(f"Training set size: {X_train.shape} && {y_train.shape}")
    print(f"Validation set size: {X_val.shape} && {y_val.shape}")
    print(f"Test set size: {X_test.shape} && {y_test.shape}")

    # Display some sample images and masks from the training set
    for i in range(5):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(X_train[i, :, :, 0], cmap='gray')
        plt.title("Image (Train)")
        plt.subplot(1, 2, 2)
        plt.imshow(y_train[i, :, :, 0], cmap='gray')
        plt.title("Mask (Train)")
        plt.show()
        plt.close()





