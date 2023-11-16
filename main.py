import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from numpy.ma.core import mean

IMAGE_FOLDER = "./Dataset/"
IMAGE_FILES = os.listdir(IMAGE_FOLDER)
TEST_FOLDER = "./Testset/"

def load_images_training_and_testing(TEST_IMAGE_PATH):
    test_img = np.asarray(Image.open(TEST_IMAGE_PATH)).flatten()
    train_imgs = []
    for img_name in IMAGE_FILES:
        train_imgs.append(np.asarray(Image.open(IMAGE_FOLDER + img_name)).flatten())
    train_imgs = np.array(train_imgs)
    return test_img, train_imgs
   
def normalize(test_img, train_imgs):
    mean_img = np.mean(train_imgs, axis=0)
    normalized_train_imgs = train_imgs - mean_img
    normalized_test_img = test_img - mean_img
    return normalized_test_img, normalized_train_imgs

def singular_value_decomposition(images):
    return np.linalg.svd(images, full_matrices=False)

def project_and_compute_weights(img, u):
    return np.multiply(img, u)

def predict(test_img, train_imgs):
    errors = np.empty(train_imgs.shape[1])
    for i in range(0, train_imgs.shape[1]):
        errors[i] = np.linalg.norm(train_imgs[:, i] - test_img)
    return np.argmin(errors)

def display_images(tested_img, predicted_img, test_file, predicted_file, result):
    fig = plt.figure()
    test_plot = fig.add_subplot(1, 2, 1)
    test_plot.set_title(f'Test Image ({test_file})', fontsize=10)
    plt.imshow(tested_img, cmap='gray')

    test_plot = fig.add_subplot(1, 2, 2)
    color = 'green' if result else 'red'
    test_plot.set_title(f'Predicted Image ({predicted_file})', fontsize=10, color=color)
    plt.imshow(predicted_img, cmap='gray')

    plt.show(block=True)

if __name__ == "__main__":
    correct_predictions = 0
    total_predictions = 0
    for TEST_FILE in os.listdir(TEST_FOLDER):
        # Load training and test images
        test_image, training_images = load_images_training_and_testing(TEST_FOLDER + TEST_FILE)

        # Normalize training and test images
        test_image, training_images = normalize(test_image, training_images)
        test_image = test_image.T
        training_images = training_images.T
        test_image = np.reshape(test_image, (test_image.size, 1))

        # Singular value decomposition
        u, _, _ = singular_value_decomposition(training_images)

        # Weight for test
        weights_test_image = project_and_compute_weights(test_image, u)
        weights_test_image = np.array(weights_test_image, dtype='int8').flatten()

        # Weights for training set
        weights_training_images = []
        for i in range(training_images.shape[1]):
            weights_i = project_and_compute_weights(np.reshape(training_images[:, i], (training_images[:, i].size, 1)), u)
            weights_i = np.array(weights_i, dtype='int8').flatten()
            weights_training_images.append(weights_i)
        weights_training_images = np.array(weights_training_images).T

        # Predict 
        index_of_most_similar_face = predict(weights_test_image, weights_training_images)

        # Showing results
        print("Test : " + TEST_FILE)
        print(f"The predicted face is: {IMAGE_FILES[index_of_most_similar_face]}")
        print("\n***************************\n")

        # Calculating Accuracy
        total_predictions += 1
        if IMAGE_FILES[index_of_most_similar_face].split("-")[0] == TEST_FILE.split("-")[0]:
            correct_predictions += 1
            # Plotting correct predictions 
            display_images(Image.open(TEST_FOLDER + TEST_FILE), Image.open(IMAGE_FOLDER + IMAGE_FILES[index_of_most_similar_face]), 
                            TEST_FILE, IMAGE_FILES[index_of_most_similar_face], True)
        else:
            # Plotting wrong predictions
            display_images(Image.open(TEST_FOLDER + TEST_FILE), Image.open(IMAGE_FOLDER + IMAGE_FILES[index_of_most_similar_face]),
                            TEST_FILE, IMAGE_FILES[index_of_most_similar_face], False)

    # Showing Accuracy
    accuracy = correct_predictions / total_predictions
    print(f'Accuracy : {"{:.2f}".format(accuracy * 100)} %')