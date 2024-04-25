import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import os
from skimage.transform import resize
from keras import models, layers
from keras import backend as K
import tensorflow as tf


# Функция для изменения размера изображений
def resize_image(image, target_shape):
    resized_image = resize(image, target_shape, anti_aliasing=True)
    return resized_image


# Функция для загрузки данных и изменения размера
def load_and_resize_data(images_path, labels_path, target_shape):
    images = []
    labels = []

    for filename in sorted(os.listdir(images_path)):
        if filename.endswith('.nii'):
            img_path = os.path.join(images_path, filename)
            img_nii = nib.load(img_path)
            img_data = img_nii.get_fdata()
            images.append(resize_image(img_data, target_shape))

    for filename in sorted(os.listdir(labels_path)):
        if filename.endswith('.nii'):
            label_path = os.path.join(labels_path, filename)
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            labels.append(resize_image(label_data, target_shape))

    return np.array(images), np.array(labels)


def unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Contraction path
    c1 = layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)

    c2 = layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)

    # Bottom of U
    c3 = layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    # Expansion path
    u4 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    u5 = layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Output layer
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model

# Функция метрики jaccard_index

def k_jaccard_index(y_true, y_pred):
    smooth = 1e-15
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# Создание и обучение различных моделей
def create_models(input_shape, num_models):
    models_list = []


    for i in range(num_models):
        model = unet_model(input_shape)
        models_list.append(model)

    return models_list


# Сравнение средних индексов Жакара всех моделей
def compare_models(models, X_test, y_test):
    jaccard_results = []

    for i, model in enumerate(models, start=1):
        print(f"Evaluating model {i}")
        scores = model.evaluate(X_test, y_test, verbose=0)
        jaccard_results.append(scores[1])
        print(f"Model {i} Jaccard Index: {scores[1]}")

    best_model_index = np.argmax(jaccard_results)
    print(f"The best model is model {best_model_index + 1} with a Jaccard Index of {jaccard_results[best_model_index]}")
    return best_model_index


# Главная функция, которая связывает весь процесс вместе
def main():
    # Задаем целевой размер изображений
    target_shape = (128, 128, 128)

    # Загрузка данных
    images_path = 'Task03_Liver_rs/imagesTr'
    labels_path = 'Task03_Liver_rs/labelsTr'
    images, labels = load_and_resize_data(images_path, labels_path, target_shape)

    # Нормализация и подготовка данных
    images = images.astype('float16') / np.max(images)
    labels = (labels > 0.5).astype('float16')

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Создание моделей
    input_shape = (128, 128, 128, 1)
    num_models = 3  # Количество моделей, которые вы хотите создать и сравнить
    models = create_models(input_shape, num_models)

    # Обучение моделей и оценка результатов
    for model in models:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', k_jaccard_index])
        model.summary()
        model.fit(X_train, y_train, validation_split=0.1, batch_size=2, epochs=5, verbose=1)

    # Сравнение моделей
    best_model_index = compare_models(models, X_test, y_test)


if __name__ == "__main__":
    main()