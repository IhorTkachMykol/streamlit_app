
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import time


# Завантаження моделей
cnn_model = load_model('cnn_model.h5')
vgg16_model = load_model('vgg16_model.h5')

# Вибір моделі
st.title('Класифікатор одягу')
st.sidebar.title('Виберіть модель')
model_type = st.sidebar.selectbox('Модель', ('CNN', 'VGG16'))

# Завантаження зображення
uploaded_file = st.file_uploader("Завантажте зображення", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Відображення зображення
    image = load_img(uploaded_file, target_size=(28, 28) if model_type == 'CNN' else (32, 32))
    st.image(image, caption='Завантажене зображення', use_column_width=True)

    # Попередня обробка зображення
    image = img_to_array(image)
    image = image.reshape((1, *image.shape))
    if model_type == 'CNN':
        image = image[:, :, :, :1]  # Залишаємо тільки один канал для ч/б зображень
    image = image.astype('float32') / 255.0

    # Передбачення
    if model_type == 'CNN':
        model = cnn_model
    else:
        model = vgg16_model

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Виведення результатів
    st.write(f'Передбачений клас: {predicted_class}')
    st.write('Ймовірності для кожного класу:')
    st.write(predictions)

# Візуалізація графіків функції втрат і точності
st.sidebar.title('Візуалізація навчання')
history_file = st.sidebar.file_uploader("Завантажте файл історії навчання (history.npz)", type=["npz"])

if history_file is not None:
    history = np.load(history_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='train_loss')
    ax1.plot(history['val_loss'], label='val_loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['accuracy'], label='train_accuracy')
    ax2.plot(history['val_accuracy'], label='val_accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    st.pyplot(fig)

st.write("Завантажте зображення для класифікації, виберіть модель та перегляньте результати.")

# Додати заголовок у додаток
st.title('cтан завантаження')

# Створити progress bar
progress = st.progress(0)

# Імітація довготривалого процесу
for i in range(100):
    # Оновлення progress bar кожну секунду
    time.sleep(0.1)  # Пауза на 0.1 секунди
    progress.progress(i + 1)  # Оновити progress bar

# Вивести повідомлення про завершення процесу
st.success('Обробка завершена!')
