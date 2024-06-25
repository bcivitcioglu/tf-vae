import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from vae_implementation import create_vae  

def preprocess_images(images):
    return images.reshape((images.shape[0], 28, 28, 1)) / 255.0

def create_dnn_classifier():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_images(vae, num_images=200):
    random_latent_points = np.random.normal(size=(num_images, vae.encoder.outputs[0].shape[1]))
    generated_images = vae.decoder.predict(random_latent_points)
    return generated_images

def evaluate_generated_images(classifier, generated_images, threshold_high=0.9, threshold_low=0.5):
    predictions = classifier.predict(generated_images)
    max_probs = np.max(predictions, axis=1)
    
    strong_predictions = max_probs > threshold_high
    weak_predictions = max_probs < threshold_low
    
    strong_pred_images = generated_images[strong_predictions]
    strong_pred_labels = np.argmax(predictions[strong_predictions], axis=1)
    
    weak_pred_images = generated_images[weak_predictions]
    weak_pred_labels = np.argmax(predictions[weak_predictions], axis=1)
    
    recognition_rate = sum(strong_predictions) / len(generated_images)
    
    return strong_pred_images, strong_pred_labels, weak_pred_images, weak_pred_labels, max_probs, recognition_rate

def plot_results(images, labels, title, num_to_show=25):
    n_images = min(num_to_show, len(images))
    rows = int(np.ceil(n_images / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(12, 2.5 * rows))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(images[i].reshape(28, 28), cmap='gray')
            ax.set_title(f"Pred: {labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confidence_histogram(confidences):
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, edgecolor='black')
    plt.title('Distribution of Prediction Confidences')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.show()

print("Loading MNIST dataset:")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)


print("Training DNN classifier:")
classifier = create_dnn_classifier()
classifier.fit(x_train, y_train, epochs=5, validation_split=0.1)

print("Training VAE:")
vae = create_vae((28, 28, 1), latent_dim=2)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=10, batch_size=128)

print("Generating new images using VAE:")
generated_images = generate_images(vae, num_images=200)

print("Evaluating generated images:")
strong_pred_images, strong_pred_labels, weak_pred_images, weak_pred_labels, confidences, recognition_rate = evaluate_generated_images(classifier, generated_images, threshold_high=0.9, threshold_low=0.5)

print(f"Recognition rate (confidence > 0.9): {recognition_rate:.2%}")
print(f"Number of strongly predicted images: {len(strong_pred_images)}")
print(f"Number of weakly predicted images: {len(weak_pred_images)}")


print("Plotting confidence distribution:")
plot_confidence_histogram(confidences)

print("Plotting a sample of strongly predicted images:")
plot_results(strong_pred_images, strong_pred_labels, "Strongly Predicted Generated Images")

if len(weak_pred_images) > 0:
    print("Plotting a sample of weakly predicted images:")
    plot_results(weak_pred_images, weak_pred_labels, "Weakly Predicted Generated Images")
else:
    print("No weakly predicted images to display.")

print("Plotting a sample of lowest confidence images:")
lowest_confidence_indices = np.argsort(confidences)[:25]
lowest_confidence_images = generated_images[lowest_confidence_indices]
lowest_confidence_labels = np.argmax(classifier.predict(lowest_confidence_images), axis=1)
plot_results(lowest_confidence_images, lowest_confidence_labels, "Lowest Confidence Generated Images")

HIGH_CONFIDENCE_THRESHOLD = 0.95  

# Select high confidence predictions
high_confidence_mask = confidences >= HIGH_CONFIDENCE_THRESHOLD
high_confidence_images = generated_images[high_confidence_mask]
high_confidence_labels = np.argmax(classifier.predict(high_confidence_images), axis=1)

print(f"Number of high-confidence generated images: {len(high_confidence_images)}")
print(f"Percentage of generated images considered trustworthy: {len(high_confidence_images)/len(generated_images):.2%}")

# Display class distribution of trusted images
unique, counts = np.unique(high_confidence_labels, return_counts=True)
print("\nClass distribution of trusted generated images:")
for digit, count in zip(unique, counts):
    print(f"Digit {digit}: {count} images")


