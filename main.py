import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

FOLDER = "./Dataset/"
FILES = os.listdir(FOLDER)
TEST_DIR = "./Testset/"

def load_images_train_and_test(TEST):
    test = np.asarray(Image.open(TEST)).flatten()
    train = [np.asarray(Image.open(FOLDER + name)).flatten() for name in FILES]
    train = np.array(train)
    return test, train

def normalize(test, train):
    mean_image = np.mean(train, axis=0)
    normalized_train = train - mean_image
    normalized_test = test - mean_image
    return normalized_test, normalized_train

def svd_function(images):
    return np.linalg.svd(images, full_matrices=False)

def project_and_calculate_weights(img, u):
    return np.multiply(img, u)

def predict(test, train):
    errors = np.linalg.norm(train - test, axis=0)
    return np.argmin(errors)

def plot_face(tested, predicted, test_file, predicted_file, result):
    f, axes = plt.subplots(1, 2)
    
    axes[0].set_title(f'Test Image ({test_file})', fontsize=10)
    axes[0].imshow(tested, cmap='gray')

    axes[1].set_title(f'Predicted Image ({predicted_file})', fontsize=10, color='green' if result else 'red')
    axes[1].imshow(predicted, cmap='gray')

    plt.show(block=True)

if __name__ == "__main__":
    true_predicts = 0
    all_predicts = 0

    for TEST_FILE in os.listdir(TEST_DIR):
        # Loading train and test
        test, train = load_images_train_and_test(TEST_DIR + TEST_FILE)

        # Normalizing train and test
        test, train = normalize(test, train)
        test = test.reshape((test.size, 1))

        # Singular value decomposition
        u, _, _ = svd_function(train.T)

        # Weights for test
        w_test = project_and_calculate_weights(test, u).flatten()

        # Weights for train set
        w_train = project_and_calculate_weights(train.T, u)

        # Predict
        index_of_most_similar_face = predict(w_test, w_train)

        # Showing results
        print("Test : " + TEST_FILE)
        print(f"The predicted face is: {FILES[index_of_most_similar_face]}")
        print("\n***************************\n")

        # Calculating Accuracy
        all_predicts += 1
        if FILES[index_of_most_similar_face].split("-")[0] == TEST_FILE.split("-")[0]:
            true_predicts += 1
            # Plotting correct predictions 
            plot_face(Image.open(TEST_DIR + TEST_FILE), Image.open(FOLDER + FILES[index_of_most_similar_face]), 
                        TEST_FILE, FILES[index_of_most_similar_face], True)
        else:
            # Plotting wrong predictions
            plot_face(Image.open(TEST_DIR + TEST_FILE), Image.open(FOLDER + FILES[index_of_most_similar_face]),
                        TEST_FILE, FILES[index_of_most_similar_face], False)

    # Showing Accuracy
    accuracy = true_predicts / all_predicts
    print(f'Accuracy: {"{:.2f}".format(accuracy * 100)}%')