import os
import glob
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import nibabel as nib  # Import nibabel to handle NIfTI (.nii) files
from tensorflow.keras import backend as K
import time
from pyswarm import pso


# Kiểm tra xem TensorFlow có nhận diện được GPU không
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/GPU:0'):  # Sử dụng GPU đầu tiên
    # Path to the ACDC dataset
    data_path = 'C:/Users/user2/Desktop/Phu1/amos22/amos22/'

    # Folders for training and testing data
    train_images_path = os.path.join(data_path, 'imagesTr')
    train_masks_path = os.path.join(data_path, 'labelsTr')
    
    test_images_path = os.path.join(data_path, 'imagesTs')
    test_masks_path = os.path.join(data_path, 'labelsTs')
    
    val_images_path = os.path.join(data_path, 'imagesVa')
    val_masks_path = os.path.join(data_path, 'labelsVa')
    
    # Lấy danh sách file paths cho ảnh và masks
    train_images = sorted(glob.glob(os.path.join(train_images_path, '*.nii.gz')))
    train_masks = sorted(glob.glob(os.path.join(train_masks_path, '*.nii.gz')))

    test_images = sorted(glob.glob(os.path.join(test_images_path, '*.nii.gz')))
    test_masks = sorted(glob.glob(os.path.join(test_masks_path, '*.nii.gz')))

    validation_images = sorted(glob.glob(os.path.join(val_images_path, '*.nii.gz')))
    validation_masks = sorted(glob.glob(os.path.join(val_masks_path, '*.nii.gz')))
    
    
    # train_images = train_images[:50]
    # train_masks = train_masks[:50]

    # test_images = test_images[:50]
    # test_masks = test_masks[:50]

    # validation_images = validation_images[:50]
    # validation_masks = validation_masks[:50]
    
    
    # Gộp tất cả ảnh và masks lại
    image_paths = train_images + validation_images + test_images
    mask_paths = train_masks + validation_masks + test_masks


    # Khởi tạo danh sách lưu ảnh và mask
    Images = []
    Masks = []
    img_size = 64  # Resize về 64x64 để tiết kiệm bộ nhớ

    def min_max_normalize(image):
        """Chuẩn hóa ảnh về dải [0,1] bằng Min-Max Scaling."""
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    # Duyệt qua từng ảnh và mask tương ứng
    for img_path, mask_path in zip(image_paths, mask_paths):
        print(f"Processing image: {img_path}")
        print(f"Processing mask: {mask_path}")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"File not found: {img_path} or {mask_path}")
            continue

        # Load ảnh và mask
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        print(f"Original Image Shape: {img.shape}")
        print(f"Original Mask Shape: {mask.shape}")

        # Tự động chọn trục Z
        z_axis = 2  # Trục Z
        num_slices = img.shape[z_axis]

        if num_slices < 4:
            print(f"Skipping {img_path} due to insufficient slices: {num_slices}")
            continue
        
        # Xác định phạm vi lát cắt từ 50% đến 100% chiều dài trục Z
        start_slice = num_slices // 2  # 50% trục Z
        end_slice = num_slices  # 100% trục Z

        # Duyệt qua tất cả các lát cắt trong khoảng từ 50% đến 100% trục Z
        slice_non_zero_counts = []

        for slice_index in range(start_slice, end_slice):
            # Trích xuất lát cắt theo trục Z
            img_slice = np.take(img, slice_index, axis=z_axis)
            mask_slice = np.take(mask, slice_index, axis=z_axis)

            # Đếm số lượng pixel khác 0 trong mask
            non_zero_count = np.count_nonzero(mask_slice)
            slice_non_zero_counts.append((slice_index, non_zero_count))

        # Sắp xếp các lát cắt theo số lượng lớp không bằng 0 giảm dần
        sorted_slices = sorted(slice_non_zero_counts, key=lambda x: x[1], reverse=True)

        # Lấy 15 lát cắt có số lớp không bằng 0 nhiều nhất
        top_15_slices = sorted_slices[:15]

        for slice_index, _ in top_15_slices:
            img_slice = np.take(img, slice_index, axis=z_axis)
            mask_slice = np.take(mask, slice_index, axis=z_axis)

            # Chuẩn hóa ảnh
            img_slice = min_max_normalize(img_slice).astype('float32')
            
            # Resize ảnh và mask
            img_resized = cv2.resize(img_slice, (img_size, img_size), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask_slice, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            
            # Xoay ảnh và mask 90 độ (counterclockwise)
            img_rotated = np.rot90(img_resized, k=1)  # Rotate by -90 degrees (counterclockwise)
            mask_rotated = np.rot90(mask_resized, k=1)  # Rotate by -90 degrees (counterclockwise)

            # Lưu vào danh sách
            Images.append(img_rotated[..., np.newaxis])  # Thêm channel dimension
            Masks.append(mask_rotated[..., np.newaxis].astype('int32'))

    print(f"Processed {len(Images)} slices.")


    # Convert lists to numpy arrays
    Images = np.array(Images)
    Masks = np.array(Masks)

    print(f"Final Images shape: {Images.shape}")
    print(f"Final Masks shape: {Masks.shape}")


    # Kiểm tra kích thước sau khi xử lý
    print(f"Final Images Shape: {Images.shape}")  # (N, 64, 64, 1)
    print(f"Final Masks Shape: {Masks.shape}")  # (N, 64, 64, 1)
    print("Unique values in masks:", np.unique(Masks))  # Kiểm tra nhãn có đúng 0,1,2,3 không
    
    
    
    def visualize_images_and_masks(Images, Masks, num_images=100):
        """
        Hiển thị một số lượng ảnh và mask nhất định (mặc định là 100).
        
        Parameters:
            Images (numpy array): Dữ liệu ảnh có dạng (N, H, W, 1).
            Masks (numpy array): Dữ liệu mask có dạng (N, H, W, 1).
            num_images (int): Số lượng ảnh và mask cần hiển thị (mặc định là 100).
        """
        # Số lượng ảnh hiển thị không lớn hơn tổng số ảnh có trong tập dữ liệu
        num_images = min(num_images, Images.shape[0])

        # Tạo figure để hiển thị các ảnh và mask
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 2))

        for i in range(num_images):
            # Hiển thị ảnh gốc
            axes[i, 0].imshow(Images[i, :, :, 0], cmap='gray')
            axes[i, 0].set_title(f"Ảnh {i+1}")
            axes[i, 0].axis('off')

            # Hiển thị mask tương ứng
            axes[i, 1].imshow(Masks[i, :, :, 0], cmap='jet', alpha=0.6)
            axes[i, 1].set_title(f"Mask {i+1}")
            axes[i, 1].axis('off')

        # Tạo khoảng cách cho các ảnh
        plt.tight_layout()
        plt.show()

    # Gọi hàm để hiển thị 100 ảnh và mask
    visualize_images_and_masks(Images, Masks, num_images=15)


    def preprocess_data(images, masks):
        """
        Preprocess the images and masks for model input.
        """
        images = np.array(images)
        masks = np.array(masks)
        
        print("Masks shape:", masks.shape)
        print("Unique values in masks before processing:", np.unique(masks))
        
        if masks.ndim == 4 and masks.shape[-1] == 1:  # Single-channel mask
            masks = np.squeeze(masks, axis=-1)  # Remove the last dimension if it's 1
            # masks = round_mask(masks)  # Round and clip mask values
            
            # Check unique mask values after rounding
            print("Unique values in masks after rounding:", np.unique(masks))
            
            # One-hot encoding for 15 classes
            num_classes = 15
            masks_one_hot = np.zeros((*masks.shape, num_classes), dtype=int)
            
            # One-hot encoding: Set the appropriate class channel to 1
            for i in range(num_classes):
                masks_one_hot[masks == i, i] = 1
            
            # Ensure the one-hot encoding is correct
            print("Unique values in one-hot encoded masks:", np.unique(masks_one_hot))
            
            masks = masks_one_hot
        
        # Split the masks into individual class layers (channels)
        masks_split = [masks[..., i] for i in range(15)]
        
        return images, masks_split


    def visualize_images_and_masks(images, masks_split, num_images=5):
        """
        Hiển thị hình ảnh và mặt nạ tương ứng cho 15 lớp.
        """
        class_names = ["Liver", "Left Kidney", "Right Kidney", "Spleen", "Pancreas", "Stomach", "Small Bowel", 
                    "Large Bowel", "Gallbladder", "Urinary Bladder", "Left Adrenal Gland", "Right Adrenal Gland", 
                    "Abdominal Aorta", "Inferior Vena Cava", "Duodenum"]
        colormaps = ['gray', 'coolwarm', 'spring', 'viridis', 'plasma', 'magma', 'cividis', 'Blues', 'Greens', 'Reds',
                    'Oranges', 'Purples', 'pink', 'YlGnBu', 'bone']
        
        for i in range(num_images):
            fig, axes = plt.subplots(1, 16, figsize=(20, 5))  # 1 hàng, 16 cột
            
            axes[0].imshow(images[i, ..., 0], cmap='gray')  # Ảnh gốc
            axes[0].set_title(f"Image {i+1}")
            axes[0].axis('off')
            
            for j in range(15):
                axes[j+1].imshow(masks_split[j][i], cmap=colormaps[j])  # Mặt nạ từng lớp
                axes[j+1].set_title(class_names[j])
                axes[j+1].axis('off')
            
            plt.tight_layout()
            plt.show()

    
    images, multi_mask = preprocess_data(Images, Masks)

    visualize_images_and_masks(images, multi_mask, num_images=15)
    
    # Convert multi_mask to a NumPy array (if not already)
    multi_mask = np.array(multi_mask)

    # Transpose the masks to have the shape (60, 64, 64, 15)
    multi_mask = np.transpose(multi_mask, (1, 2, 3, 0))

    # Stack the masks (no need to do np.stack anymore)
    masks = multi_mask

    # Print the shapes after stacking
    print('AFTER_Images_shape', images.shape)
    print('AFTER_Masks_shape', masks.shape)
    
    # Chia tập train (70%), tập còn lại (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(images, masks, train_size=0.7, random_state=42)

    # Chia tiếp tập validation (10%) và test (20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=1/3, test_size=2/3, random_state=42)

    # In kích thước của các tập dữ liệu
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Tính một-hot encoding cho ground truth (y_true)
    def dice_score(y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Nếu cần, làm mềm (smooth) giá trị để tránh chia cho 0
        smooth = 1e-5
        intersection = K.sum(y_true * y_pred)
        y_true_sum = K.sum(y_true)
        y_pred_sum = K.sum(y_pred)

        dice = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
        return dice
    
    def hausdorff_distance(y_true, y_pred, max_distance=None):
        """
        Computes the Hausdorff distance at the 95th percentile between the ground truth points (y_true)
        and predicted points (y_pred), and scales the result to be between 0 and 1.
        
        Parameters:
        - y_true: Tensor of shape (N, D), where N is the number of points and D is the dimensionality.
        - y_pred: Tensor of shape (M, D), where M is the number of points and D is the dimensionality.
        - max_distance: The maximum possible distance in the dataset to scale the Hausdorff distance. If None, it will be calculated.
        
        Returns:
        - The scaled Hausdorff distance at the 95th percentile.
        """
        # Calculate the pairwise distances between points in y_true and y_pred
        diff = tf.expand_dims(y_true, 1) - tf.expand_dims(y_pred, 0)
        pairwise_distances = tf.norm(diff, axis=-1, ord='euclidean')  # L2 norm (Euclidean distance)

        # For each point in y_true, find the closest point in y_pred (infimum of distances)
        min_dist_true_to_pred = tf.reduce_min(pairwise_distances, axis=1)
        
        # For each point in y_pred, find the closest point in y_true (infimum of distances)
        min_dist_pred_to_true = tf.reduce_min(pairwise_distances, axis=0)
        
        # Combine both sets of minimum distances
        all_distances = tf.concat([min_dist_true_to_pred, min_dist_pred_to_true], axis=0)
        
        # Compute the 95th percentile using tf.percentile
        hd95 = tf.reduce_mean(tf.sort(all_distances))  # Sort distances and compute the 95th percentile manually
        # index = int(0.95 * len(all_distances))
        index = int(0.95 * tf.cast(len(all_distances), tf.float32))  # Ensure the index is an integer
        hd95 = all_distances[index]
        
        # If max_distance is not provided, calculate it based on the range of distances in the dataset
        if max_distance is None:
            max_distance = tf.reduce_max(pairwise_distances)
        
        # Scale the Hausdorff distance to be between 0 and 1
        scaled_hd95 = hd95 / max_distance
        
        return scaled_hd95
    
    def dice_loss(y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Nếu cần, làm mềm (smooth) giá trị để tránh chia cho 0
        smooth = 1e-5
        intersection = K.sum(y_true * y_pred)
        y_true_sum = K.sum(y_true)
        y_pred_sum = K.sum(y_pred)

        dice = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
        return 1 - dice  # Dice loss là 1 - Dice coefficient

    
    # Hàm mất mát tổ hợp
    def combined_loss(alpha, beta):
        def loss_fn(y_true, y_pred):
            # Tính Dice Loss
            dice = dice_loss(y_true, y_pred)
            # Tính Binary Cross-Entropy Loss
            cross_entropy = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
            # Ensure all tensors are float32
            dice = tf.cast(dice, tf.float32)
            cross_entropy = tf.cast(cross_entropy, tf.float32)
            # Kết hợp hai hàm mất mát
            return alpha * dice + beta * cross_entropy
        return loss_fn


    def unet_model(input_shape=(64, 64, 1), num_classes=15, learning_rate=0.0005, dropout_rate=0.3, filter_base=32, filter_growth=2, kernel_size=3, alpha=0.5, beta=0.5, activation='relu'):

        inputs = Input(shape=input_shape)
        
        # Contracting path
        c1 = Conv2D(filter_base, (kernel_size,kernel_size), activation=activation, padding='same')(inputs)
        c1 = Conv2D(filter_base, (kernel_size,kernel_size), activation=activation, padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout_rate)(p1)
        
        c2 = Conv2D(int(filter_base * filter_growth), (kernel_size,kernel_size), activation=activation, padding='same')(p1)
        c2 = Conv2D(int(filter_base * filter_growth), (kernel_size,kernel_size), activation=activation, padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout_rate)(p2)
        
        c3 = Conv2D(int(filter_base * filter_growth**2), (kernel_size,kernel_size), activation=activation, padding='same')(p2)
        c3 = Conv2D(int(filter_base * filter_growth**2), (kernel_size,kernel_size), activation=activation, padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout_rate)(p3)
        
        c4 = Conv2D(int(filter_base * filter_growth**3), (kernel_size,kernel_size), activation=activation, padding='same')(p3)
        c4 = Conv2D(int(filter_base * filter_growth**3), (kernel_size,kernel_size), activation=activation, padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout_rate)(p4)
        
        # Bottleneck
        c5 = Conv2D(int(filter_base * filter_growth**4), (kernel_size,kernel_size), activation=activation, padding='same')(p4)
        c5 = Conv2D(int(filter_base * filter_growth**4), (kernel_size,kernel_size), activation=activation, padding='same')(c5)
        
        # Expanding path
        u6 = UpSampling2D((2, 2))(c5)
        u6 = Concatenate()([u6, c4])
        c6 = Conv2D(int(filter_base * filter_growth**3), (kernel_size,kernel_size), activation=activation, padding='same')(u6)
        c6 = Conv2D(int(filter_base * filter_growth**3), (kernel_size,kernel_size), activation=activation, padding='same')(c6)
        
        u7 = UpSampling2D((2, 2))(c6)
        u7 = Concatenate()([u7, c3])
        c7 = Conv2D(int(filter_base * filter_growth**2), (kernel_size,kernel_size), activation=activation, padding='same')(u7)
        c7 = Conv2D(int(filter_base * filter_growth**2), (kernel_size,kernel_size), activation=activation, padding='same')(c7)
        
        u8 = UpSampling2D((2, 2))(c7)
        u8 = Concatenate()([u8, c2])
        c8 = Conv2D(int(filter_base * filter_growth), (kernel_size,kernel_size), activation=activation, padding='same')(u8)
        c8 = Conv2D(int(filter_base * filter_growth), (kernel_size,kernel_size), activation=activation, padding='same')(c8)
        
        u9 = UpSampling2D((2, 2))(c8)
        u9 = Concatenate()([u9, c1])
        c9 = Conv2D(filter_base, (kernel_size,kernel_size), activation=activation, padding='same')(u9)
        c9 = Conv2D(filter_base, (kernel_size,kernel_size), activation=activation, padding='same')(c9)
        
        outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
        
        model = Model(inputs, outputs)
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                    loss=combined_loss(alpha=alpha, beta=beta),
                    metrics=['accuracy', dice_score, hausdorff_distance])
        
        return model

    # Clear session to release GPU memory
    K.clear_session()

    # 2. Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Fitness function for PSO
    def fitness_function(params):
        learning_rate = params[0]
        dropout_rate = params[1]
        filter_base = int(params[2])
        filter_growth = int(params[3])
        kernel_size = int(params[4])
        activations = int(params[5]) # 0-->11
        alpha=params[6]
        beta=params[7]
        
        activation_functions = [
            'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 
            'elu', 'selu', 'gelu', 'swish', 'hard_sigmoid', 'exponential'
        ]


        print(f"Learning_Rate={learning_rate}, Dropout_Rate={dropout_rate}, Filter_Base={filter_base}, Filter_Growth={filter_growth}, Kernel_Size={kernel_size}, Alpha={alpha}, Beta={beta}, Activation={activation_functions[activations]}")

        model = unet_model(learning_rate=learning_rate, dropout_rate=dropout_rate, filter_base=filter_base, filter_growth=filter_growth, kernel_size=kernel_size, alpha=alpha, beta=beta, activation=activation_functions[activations])
        
        # Start the timer
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=100, batch_size=4,
                            validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])
        # Record the training time
        training_time = time.time() - start_time
        val_loss = history.history['val_loss'][-1]
        train_loss = history.history['loss'][-1]
        # Calculate model complexity (number of parameters)
        model_complexity = model.count_params()
        print(f"Validation Loss: {val_loss}, Training Loss: {train_loss}, "
            f"Training Time: {training_time}, Model Complexity: {model_complexity}")
        
        return val_loss + train_loss

    # Define the lower bounds (lb) and upper bounds (ub) for each hyperparameter in arrays
    
    # Optimized Lower bounds array (lb)
    lb = [0.001, 0.2, 32, 1, 2, 0, 0.5, 0.1]

    # Optimized Upper bounds array (ub)
    ub = [0.01, 0.5, 64, 3, 4, 11, 0.6, 0.5]


    # Print the updated bounds
    print("Updated Lower Bounds Array:", lb)
    print("Updated Upper Bounds Array:", ub)


    # Run PSO
    # best_params, best_loss = pso(fitness_function, lb, ub, swarmsize=10, maxiter=3)
    
    # Options for the PSO
    options = {
        'swarm_size': 5,  # Number of particles
        'maxiter': 10,    # Maximum number of iterations
        'debug': True,     # Optional: print debugging information
    }

    # Run PSO with the specified options
    best_params, best_loss = pso(fitness_function, lb, ub, swarmsize=options['swarm_size'], maxiter=options['maxiter'], debug=options['debug'])

    activation_functions = [
            'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 
            'elu', 'selu', 'gelu', 'swish', 'hard_sigmoid', 'exponential'
        ]
    
    # Display best parameters
    print("\nBest Parameters Found:")
    print(f"Learning_Rate: {best_params[0]}")
    print(f"Dropout_Rate: {best_params[1]}")
    print(f"Filter_Base: {int(best_params[2])}")
    print(f"Filter_Growth: {int(best_params[3])}")
    print(f"Kernel_Size: {int(best_params[4])}")
    print(f"Activation: {activation_functions[int(best_params[5])]}")
    print(f"Alpha: {best_params[6]}")
    print(f"Beta: {best_params[7]}")
    
    
    model = unet_model(learning_rate=best_params[0], dropout_rate=best_params[1], filter_base=int(best_params[2]), filter_growth=int(best_params[3]), kernel_size=int(best_params[4]), alpha=best_params[6], beta=best_params[7], activation=activation_functions[int(best_params[5])])
    
    model.summary()

    history = model.fit(X_train, y_train, epochs=100, batch_size=4,
                             validation_data=(X_test, y_test))
    
    dice_confficent = history.history['dice_score'][-1]
    print("Dice Confficent:", dice_confficent)
    
    hausdorff_distance = history.history['hausdorff_distance'][-1]
    print("Hausdorff Distance:", hausdorff_distance)
    
    accuracy = history.history['accuracy'][-1]
    print("Accuracy:", accuracy)

    # # Tạo mô hình và biên dịch
    # model = unet_model()
    # model.summary()

    # # Huấn luyện mô hình
    # epochs = 10
    # batch_size = 8

    # history = model.fit(X_train, y_train,
    #                     validation_data=(X_val, y_val),
    #                     epochs=epochs,
    #                     batch_size=batch_size)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Number of images to visualize
    num_images = 3

    # Plotting the images, ground truth, and predicted masks on the original image
    fig, axs = plt.subplots(num_images, 3, figsize=(15, num_images * 5))

    for i in range(num_images):
        # Select an example image and mask from the test set
        image = X_test[i]
        ground_truth_mask = y_test[i]
        predicted_mask = predictions[i]

        # Assuming ground_truth_mask and predicted_mask are NumPy arrays
        gt_overlay = np.argmax(ground_truth_mask, axis=-1)
        pred_overlay = np.argmax(predicted_mask, axis=-1)

        # Print the shape
        print("Ground truth overlay shape:", gt_overlay.shape)
        print("Predicted overlay shape:", pred_overlay.shape)

        # Display the original image
        axs[i, 0].imshow(image[..., 0], cmap='gray')
        axs[i, 0].set_title(f'Image {i+1}')
        axs[i, 0].axis('off')

        # Display the ground truth mask overlay
        axs[i, 1].imshow(image[..., 0], cmap='gray')
        axs[i, 1].imshow(gt_overlay, cmap='tab10', alpha=0.5)  # Overlay with transparency
        axs[i, 1].set_title(f'Ground Truth {i+1}')
        axs[i, 1].axis('off')

        # Display the predicted mask overlay
        axs[i, 2].imshow(image[..., 0], cmap='gray')
        axs[i, 2].imshow(pred_overlay, cmap='tab10', alpha=0.5)  # Overlay with transparency
        axs[i, 2].set_title(f'Predicted {i+1}')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
    
    def dice_coef_per_class(y_true, y_pred, class_names):
        """
        Tính Dice Score cho từng lớp
        """
        dice_scores = {}
        for i, class_name in enumerate(class_names):
            dice_scores[class_name] = dice_score(y_true[..., i], y_pred[..., i])
        return dice_scores

    # Danh sách tên các lớp
    class_names = ["Liver", "Left Kidney", "Right Kidney", "Spleen", "Pancreas", "Stomach", "Small Bowel", 
                "Large Bowel", "Gallbladder", "Urinary Bladder", "Left Adrenal Gland", "Right Adrenal Gland", 
                "Abdominal Aorta", "Inferior Vena Cava", "Duodenum"]

    # Giả sử y_true và y_pred là numpy array có kích thước (N, H, W, 15)
    y_true = np.array(y_test)  # Ground truth
    y_pred = model.predict(X_test)  # Dự đoán từ model

    # Chuyển y_pred thành binary mask (lớn nhất = 1, còn lại = 0)
    y_pred = (y_pred > 0.5).astype(np.float32)

    # Tính Dice Score cho từng lớp
    dice_scores = dice_coef_per_class(y_true, y_pred, class_names)

    # In kết quả
    for class_name, dice_value in dice_scores.items():
        print(f"Dice Score - {class_name}: {dice_value:.4f}")
        
