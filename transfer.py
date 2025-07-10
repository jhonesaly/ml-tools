import os
import random
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt

# Diagnóstico completo da GPU
print("=== DIAGNÓSTICO GPU ===")
print("TensorFlow version:", tf.__version__)
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

# Tentar obter informações de build (pode falhar em versões CPU-only)
try:
    build_info = tf.sysconfig.get_build_info()
    print("CUDA version:", build_info.get('cuda_version', 'N/A'))
    print("cuDNN version:", build_info.get('cudnn_version', 'N/A'))
except Exception as e:
    print("Erro ao obter build info:", e)

# Verificar dispositivos físicos
physical_devices = tf.config.list_physical_devices()
print("Todos os dispositivos físicos:", physical_devices)

gpu_devices = tf.config.list_physical_devices('GPU')
print("GPU disponível:", gpu_devices)
print("Usando GPU:", len(gpu_devices) > 0)

# Teste básico da GPU se disponível
if len(gpu_devices) > 0:
    print("GPU detectada! Testando operação...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("Teste GPU bem-sucedido!")
    except Exception as e:
        print(f"Erro no teste GPU: {e}")

# Configurar uso de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar crescimento de memória
        for gpu in gpus:
            tf.config.experimental.set_gpu_growth(gpu, True)
        print(f"Usando GPU: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print("Nenhuma GPU detectada, usando CPU")


# Helper function to load image and return it and input vector
def get_image(path, target_size=(224, 224)):
    """Loads an image, resizes it, and preprocesses it for model input."""
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# Function to load data from a local directory structure
def load_data(
        root_dir,
        exclude_categories=None,
        train_split=0.7,
        val_split=0.15
        ):
    """
    Loads images from a directory structure where each subdirectory is a class.

    Args:
        root_dir (str): The root directory containing class subdirectories.
        exclude_categories (list, optional): A list of category names to
        exclude. train_split (float): The proportion of data to use for
        training.val_split (float): The proportion of data to use for
        validation.

    Returns:
        tuple: A tuple containing:
            - x_train (np.ndarray): Training data.
            - y_train (np.ndarray): Training labels (one-hot encoded).
            - x_val (np.ndarray): Validation data.
            - y_val (np.ndarray): Validation labels (one-hot encoded).
            - x_test (np.ndarray): Test data.
            - y_test (np.ndarray): Test labels (one-hot encoded).
            - categories (list): A list of category names.
    """
    data = []
    categories = [x[0] for x in os.walk(root_dir) if x[0]][1:]
    if exclude_categories:
        categories = [
            c for c in categories if os.path.basename(c)
            not in exclude_categories
            ]

    categories.sort()  # Ensure consistent ordering

    for c, category in enumerate(categories):
        images = [os.path.join(dp, f) for dp, dn, filenames
                  in os.walk(category) for f in filenames
                  if os.path.splitext(f)[1].lower() in [
                      '.jpg',
                      '.png',
                      '.jpeg'
                      ]]
        for img_path in images:
            try:
                img, x = get_image(img_path)
                data.append({'x': np.array(x[0]), 'y': c})
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

    random.shuffle(data)

    idx_val = int(train_split * len(data))
    idx_test = int((train_split + val_split) * len(data))
    train = data[:idx_val]
    val = data[idx_val:idx_test]
    test = data[idx_test:]

    x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
    x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
    x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]

    num_classes = len(categories)

    # Normalize data
    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Convert labels to one-hot vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(f"Finished loading {len(data)} images from {num_classes} categories")
    print(f"Train / validation / test split: {len(x_train)}, {len(x_val)}, {len(x_test)}")
    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, categories


# Function to build and train the transfer learning model
def build_and_train_transfer_model(
        x_train,
        y_train,
        x_val,
        y_val,
        num_classes,
        epochs=10,
        batch_size=128
        ):
    """
    Builds and trains a transfer learning model based on VGG16.

    Args:
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels (one-hot encoded).
        x_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels (one-hot encoded).
        num_classes (int): The number of output classes.
        epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.

    Returns:
        keras.models.Model: The trained transfer learning model.
        keras.callbacks.History: The training history.
    """
    # Load VGG16 pre-trained on ImageNet
    vgg = keras.applications.VGG16(weights='imagenet', include_top=True)

    # Create a new model by taking VGG up to the second to last layer
    inp = vgg.input
    new_classification_layer = Dense(num_classes, activation='softmax')
    out = new_classification_layer(vgg.layers[-2].output)
    model_new = Model(inp, out)

    # Freeze all layers except the new classification layer
    for layer in model_new.layers[:-1]:
        layer.trainable = False
    for layer in model_new.layers[-1:]:
        layer.trainable = True

    # Compile the model
    model_new.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    model_new.summary()

    # Train the model
    history = model_new.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val))

    return model_new, history


# Function to evaluate the model
def evaluate_model(model, x_test, y_test):
    """Evaluates the trained model on the test set."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)


# Function to plot training history
def plot_training_history(history):
    """Plots the validation loss and accuracy over epochs."""
    fig = plt.figure(figsize=(16,4))
    ax = fig.add_subplot(121)
    ax.plot(history.history["val_loss"])
    ax.set_title("validation loss")
    ax.set_xlabel("epochs")

    ax2 = fig.add_subplot(122)
    # Compatibilidade com versões antigas e novas do Keras
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    ax2.plot(history.history[val_acc_key])
    ax2.set_title("validation accuracy")
    ax2.set_xlabel("epochs")
    ax2.set_ylim(0, 1)

    plt.show()


# Function to load trained model
def load_trained_model(model_path):
    """Load a previously trained model."""
    if os.path.exists(model_path):
        print(f"Carregando modelo salvo: {model_path}")
        return keras.models.load_model(model_path)
    return None


# Main execution block
if __name__ == "__main__":
    # Define your data root directory and categories to exclude (if any)
    # Replace 'your_dataset_root' with the path to your image dataset
    # Your dataset should be organized with class subdirectories
    data_root = 'dataset' # Replace with your dataset path
    excluded_categories = [] # Replace with categories to exclude

    # Load and preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test, categories = load_data(
        data_root, exclude_categories=excluded_categories
    )

    # Verificar se já existe um modelo treinado
    model_filename = 'transfer_learning_model.h5'
    transfer_model = load_trained_model(model_filename)

    if transfer_model is None:
        print("Nenhum modelo encontrado. Iniciando treinamento...")
        # Build and train the transfer learning model
        transfer_model, transfer_history = build_and_train_transfer_model(
            x_train,
            y_train,
            x_val,
            y_val,
            num_classes=len(categories),
            epochs=10
        )

        # Salvar o modelo treinado
        transfer_model.save(model_filename)
        print(f"Modelo salvo como: {model_filename}")

        # Plot transfer learning training history
        plot_training_history(transfer_history)
    else:
        print("Modelo carregado com sucesso! Pulando treinamento...")
        transfer_history = None

    # Evaluate the transfer learning model
    print("\nEvaluating Transfer Learning Model:")
    evaluate_model(transfer_model, x_test, y_test)

    # Plot training history (apenas se houver histórico)
    if transfer_history is not None:
        plot_training_history(transfer_history)

    # Example prediction (replace 'path/to/your/image.jpg' with a valid image path)
    # Make sure the image path is within one of the class subdirectories
    # For example, to predict an image from the 'anchor' category:
    # pred_image_path = os.path.join(data_root, 'anchor', 'image_0001.jpg')
    # try:
    #     img_to_predict, x_to_predict = get_image(pred_image_path)
    #     probabilities = transfer_model.predict([x_to_predict])
    #     predicted_class_index = np.argmax(probabilities)
    #     predicted_category = categories[predicted_class_index]
    #     print(f"\nPrediction for {pred_image_path}:")
    #     print(f"Predicted class: {predicted_category} (Probability: {probabilities[0][predicted_class_index]:.4f})")
    # except Exception as e:
    #     print(f"Error predicting image {pred_image_path}: {e}")
