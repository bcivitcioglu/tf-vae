import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from vae_implementation import create_vae


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def train_vae(dataset, epochs=10, batch_size=128):
    vae = create_vae((28, 28, 1), latent_dim=2)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(dataset, epochs=epochs, batch_size=batch_size)
    return vae

def plot_latent_space(vae, dataset, n=30, figsize=15):
    # Display a 2D manifold of the digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # Construct grid of latent space coordinates
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

def run_demo(dataset_name):
    print(f"Running demo for {dataset_name}")
    if dataset_name == "mnist":
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset name. Choose 'mnist' or 'fashion_mnist'")

    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(128)

    vae = train_vae(train_dataset)
    plot_latent_space(vae, x_test)

run_demo("mnist")
run_demo("fashion_mnist")