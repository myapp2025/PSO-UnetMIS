import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, MultiHeadAttention, LayerNormalization,
    Add, GlobalAveragePooling1D, Reshape, Concatenate, Dropout
)
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, Activation, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
from scipy.spatial.distance import cdist
from skimage import measure
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label, regionprops

import numpy as np
from tensorflow.keras import backend as K
from pyswarm import pso
import torch
import torch.nn as nn
from medpy import metric

# Kiểm tra xem TensorFlow có nhận diện được GPU không
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/GPU:0'):  # Sử dụng GPU đầu tiên

    def dice_coefficient(y_true, y_pred):
        
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


    # Sensitivity (True Positive Rate)
    def sensitivity(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        TP = K.sum(y_true * y_pred)
        FN = K.sum((1 - y_true) * y_pred)
        return TP / (TP + FN + epsilon)  # No multiplication by 100


    # Specificity (True Negative Rate)
    def specificity(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        TN = K.sum((1 - y_true) * (1 - y_pred))
        FP = K.sum(y_true * (1 - y_pred))
        return TN / (TN + FP + epsilon)  # No multiplication by 100


    # Precision
    def precision(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        TP = K.sum(y_true * y_pred)
        FP = K.sum((1 - y_true) * y_pred)
        return TP / (TP + FP + epsilon)  # No multiplication by 100


    # Recall (True Positive Rate)
    def recall(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        TP = K.sum(y_true * y_pred)
        FN = K.sum((1 - y_true) * y_pred)
        return TP / (TP + FN + epsilon)  # No multiplication by 100


    # F1-Score
    def f1_score(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2 * (p * r) / (p + r + K.epsilon())  # No multiplication by 100


    # Matthews Correlation Coefficient (MCC)
    def mcc(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        TP = K.sum(y_true * y_pred)
        TN = K.sum((1 - y_true) * (1 - y_pred))
        FP = K.sum((1 - y_true) * y_pred)
        FN = K.sum(y_true * (1 - y_pred))

        numerator = TP * TN - FP * FN
        denominator = K.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        return numerator / (denominator + epsilon)  # No multiplication by 100


    # False Positive Rate (FPR)
    def false_positive_rate(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        FP = K.sum((1 - y_true) * y_pred)
        TN = K.sum((1 - y_true) * (1 - y_pred))
        return FP / (FP + TN + epsilon)  # No multiplication by 100


    # False Negative Rate (FNR)
    def false_negative_rate(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        FN = K.sum(y_true * (1 - y_pred))
        TP = K.sum(y_true * y_pred)
        return FN / (FN + TP + epsilon)  # No multiplication by 100
    
    
    # Tính một-hot encoding cho ground truth (y_true)
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



    # Display available devices
    print("Available devices:", tf.config.list_physical_devices())


    # def unet_model(input_shape=(64, 64, 1), num_classes=1, learning_rate=0.0005, dropout_rate=0.3):
    #     inputs = Input(shape=input_shape)
        
    #     # Contracting path
    #     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    #     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    #     p1 = MaxPooling2D((2, 2))(c1)
    #     p1 = Dropout(dropout_rate)(p1)
        
    #     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    #     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    #     p2 = MaxPooling2D((2, 2))(c2)
    #     p2 = Dropout(dropout_rate)(p2)
        
    #     c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    #     c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    #     p3 = MaxPooling2D((2, 2))(c3)
    #     p3 = Dropout(dropout_rate)(p3)
        
    #     c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    #     c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    #     p4 = MaxPooling2D((2, 2))(c4)
    #     p4 = Dropout(dropout_rate)(p4)
        
    #     # Bottleneck
    #     c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    #     c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        
    #     # Expanding path
    #     u6 = UpSampling2D((2, 2))(c5)
    #     u6 = Concatenate()([u6, c4])
    #     c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    #     c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        
    #     u7 = UpSampling2D((2, 2))(c6)
    #     u7 = Concatenate()([u7, c3])
    #     c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    #     c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        
    #     u8 = UpSampling2D((2, 2))(c7)
    #     u8 = Concatenate()([u8, c2])
    #     c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    #     c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        
    #     u9 = UpSampling2D((2, 2))(c8)
    #     u9 = Concatenate()([u9, c1])
    #     c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    #     c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        
    #     outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
        
    #     model = Model(inputs, outputs)
        
    #     model.compile(optimizer=Adam(learning_rate=learning_rate), 
    #                 loss=combined_loss(alpha=0.7, beta=0.3),
    #                 metrics=['accuracy', dice_coefficient, hausdorff_distance,
    #                         sensitivity, specificity, precision, recall,
    #                         f1_score, mcc, false_positive_rate, false_negative_rate])
        
    #     return model
    
    def unet_model(input_shape=(64, 64, 1), num_classes=1, learning_rate=0.0005, dropout_rate=0.3, filter_base=64, filter_growth=2, kernel_size=3, alpha=0.5, beta=0.5, activation='relu'):
        
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
                    metrics=['accuracy', dice_coefficient, hausdorff_distance,
                            sensitivity, specificity, precision, recall,
                            f1_score, mcc, false_positive_rate, false_negative_rate])
        
        return model

    # Clear session to release GPU memory
    K.clear_session()

    import os
    import glob
    import numpy as np
    import cv2
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Đường dẫn đến dữ liệu PH2
    data_path = 'C:/Users/user2/Desktop/Phu1/PH2_dataset'

    # Thư mục chứa ảnh và nhãn/mask
    image_folder = os.path.join(data_path, 'trainx')
    mask_folder = os.path.join(data_path, 'trainy')

    # Lấy danh sách file paths cho ảnh và masks (giả sử ảnh là BMP)
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.bmp')))  # Lấy tất cả ảnh trong thư mục
    mask_paths = sorted(glob.glob(os.path.join(mask_folder, '*.bmp')))  # Giả sử có mask dưới dạng BMP

    # Đảm bảo số lượng file ảnh và masks khớp nhau
    assert len(image_paths) == len(mask_paths), "Số lượng ảnh và masks không khớp."

    # Khởi tạo danh sách để lưu trữ dữ liệu
    Images = []
    Masks = []
    img_size = 64  # Giảm độ phân giải xuống 64x64 để tiết kiệm bộ nhớ

    # Xử lý dữ liệu
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Đọc ảnh từ file BMP (ảnh grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Đọc mask (nếu có)

        # Resize ảnh và mask về kích thước img_size x img_size
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float32') / 255.0
        mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float32') / 255.0

        Images.append(img_resized[..., np.newaxis])  # Thêm chiều thứ 3 cho dữ liệu ảnh
        Masks.append(mask_resized[..., np.newaxis])  # Thêm chiều thứ 3 cho dữ liệu mask

    # Chuyển đổi danh sách sang mảng numpy
    Images = np.array(Images)
    Masks = np.array(Masks)

    # Đảm bảo số lượng dữ liệu là 200
    assert len(Images) == 200, f"Số lượng ảnh không đúng: {len(Images)}"

    # Chia dữ liệu thành các tập con theo yêu cầu (80 cho train, 20 cho validation, 100 cho test)
    X_train, X_temp, y_train, y_temp = train_test_split(Images, Masks, test_size=0.6, random_state=42)  # 60% cho validation + test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Chia đều giữa validation và test

    # In thông tin về kích thước của các tập dữ liệu
    print(f"Số lượng ảnh train: {X_train.shape}")
    print(f"Số lượng ảnh validation: {X_val.shape}")
    print(f"Số lượng ảnh test: {X_test.shape}")

    # Hiển thị một vài mẫu ảnh và mask từ tập huấn luyện
    for i in range(5):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(X_train[i, :, :, 0], cmap='gray')
        plt.title("Image (Train)")
        plt.subplot(1, 2, 2)
        plt.imshow(y_train[i, :, :, 0], cmap='gray')
        plt.title("Mask (Train)")
        plt.show()
        plt.close()  # Close the figure to free up resources
        
    import time

    # Reshape data to match input image dimensions for pixel-wise classification
    X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
    X_val = X_val.reshape(X_val.shape[0], img_size, img_size, 1)
    X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)

    y_train = y_train.reshape(y_train.shape[0], img_size, img_size, 1)
    y_val = y_val.reshape(y_val.shape[0], img_size, img_size, 1)
    y_test = y_test.reshape(y_test.shape[0], img_size, img_size, 1)

    # 2. Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    # Define input shape
    input_shape = X_train.shape[1:]  # e.g., (64, 64, 1) for grayscale image
    
    
     # Fitness function for PSO
    def fitness_function(params):
        # input_shape=(64, 64, 1), num_classes=1, learning_rate=0.0005, dropout_rate=0.3, 
        #       filter_base=64, filter_growth=2, kernel_size=3, activation='relu'
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
        'swarm_size': 50,  # Number of particles
        'maxiter': 100,    # Maximum number of iterations
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
    # model = cnn_model()
    # model = segnet()
    
    model.summary()

    history = model.fit(X_train, y_train, epochs=100, batch_size=4,
                             validation_data=(X_test, y_test))
    
    # In kết quả cuối cùng của từng metric
    print("\nTraining finished!")
    for metric in history.history.keys():
        final_value = history.history[metric][-1]
        print(f"Final {metric}: {final_value:.4f}")

    # history = model.fit(X_train, y_train, epochs=100, batch_size=int(best_params[5]),
    #                         validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    
    import matplotlib.pyplot as plt
    import math  # Import math at the beginning

    # ... your existing code ...

    # Predict on the desired image (index 100)
    image_index = 50  # Change this to an index between 0 and 59
    predicted_mask = model.predict(X_test[image_index].reshape(1, img_size, img_size, 1))

    # Create a figure with appropriate size
    fig = plt.figure(figsize=(18, 15))

    # Original CT Image (X_test)
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[image_index][..., 0], cmap='bone')
    plt.title('Original Image')

    # Original Infection Mask (y_test)
    plt.subplot(1, 3, 2)
    plt.imshow(X_test[image_index][..., 0], cmap='bone')
    plt.imshow(y_test[image_index][..., 0], alpha=0.5, cmap="nipy_spectral")
    plt.title('Original Mask')

    # Predicted Infection Mask
    plt.subplot(1, 3, 3)
    plt.imshow(X_test[image_index][..., 0], cmap='bone')
    plt.imshow(predicted_mask[0][..., 0], alpha=0.5, cmap="nipy_spectral")  # predicted_mask[0] for the single image
    plt.title('Predicted Mask')

    plt.show()
    plt.close()  # Close the figure to free up resources


    # Plotting Training History
    def plot_training_history(history):
        """Plots the training and validation loss and accuracy."""
        # Extract metrics
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Create subplots for loss and accuracy
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Loss Plot
        ax[0].plot(loss, label='Training Loss', color='blue')
        ax[0].plot(val_loss, label='Validation Loss', color='orange')
        ax[0].set_title('Loss Over Epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Accuracy Plot
        ax[1].plot(accuracy, label='Training Accuracy', color='blue')
        ax[1].plot(val_accuracy, label='Validation Accuracy', color='orange')
        ax[1].set_title('Accuracy Over Epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    # Call the function to plot
    plot_training_history(history)

    # Evaluate the model on the test set
    # test_loss, test_accuracy, *metrics = best_model.evaluate(X_test, y_test, verbose=0, batch_size=int(best_params[5]))
    test_loss, test_accuracy, *metrics = model.evaluate(X_test, y_test, verbose=0, batch_size=4)

    # Print evaluation results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Optionally, you can also print other metrics like dice coefficient or specificity if defined in the model.

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

    # --- ROC Curve ---
    def plot_roc_curve(y_true, y_pred):
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        y_true_binary = (y_true > 0.5).astype(int)

        # Use raw predictions for ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred)  
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Precision-Recall Curve ---
    def plot_precision_recall_curve(y_true, y_pred):
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        y_true_binary = (y_true > 0.5).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='orange', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Confusion Matrix ---
    def plot_confusion_matrix(y_true, y_pred, threshold=0.5):
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        y_pred_labels = (y_pred > threshold).astype(int)
        y_true_binary = (y_true > 0.5).astype(int)

        cm = confusion_matrix(y_true_binary, y_pred_labels)
        
        # Dynamically get labels from unique values
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true_binary))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    # Get predictions for the entire test set
    y_pred_all = model.predict(X_test, batch_size=4).flatten()
    # y_pred_all = best_model.predict(X_test, batch_size=int(best_params[5]))# Predict on the entire X_test

    # Plot results
    plot_roc_curve(y_test, y_pred_all)
    plot_precision_recall_curve(y_test, y_pred_all)
    plot_confusion_matrix(y_test, y_pred_all)

