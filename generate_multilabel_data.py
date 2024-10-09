import numpy as np
import pickle
from utils.get_data import ContextGenerator
import argparse
# import gym
# import d4rl
from sklearn.datasets import load_svmlight_file
from icecream import ic
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import os
from PIL import Image
import cv2
from sklearn.utils import shuffle

# Function to load dataset
def load_scene_classification_dataset(filepath):
    # Load dataset in LIBSVM format
    data, labels = load_svmlight_file(filepath, multilabel=True)

    # Convert sparse matrix to dense format (optional)
    data_dense = data.toarray()

    return data_dense, labels

def scaleImage(x):          # Pass a PIL image, return a tensor
    y = x
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min())
    z = y - y.mean()        # Subtract the mean value of the image
    return z

def generate_data(T, n_actions, n_features, noise_std, seed, data_name='MedNIST', ):
    '''
    data_name='MedNIST', 'yeast'
    '''
    # Set random seed for reproducibility
    np.random.seed(seed)
    # env = gym.make('maze2d-umaze-v1')
    # print('obs_space: ', env.observation_space)
    # print('action_space: ', env.action_space)
    if data_name == 'yeast':
        train_data_path = 'raw_data/multilabel_ds/yeast_train.svm'
        test_data_path = 'raw_data/multilabel_ds/yeast_test.svm'
        x_train, y_train = load_scene_classification_dataset(train_data_path)
        x_test, y_test = load_scene_classification_dataset(test_data_path)
        num_classes = 14
        x_train = PCA(n_components=6).fit_transform(x_train)
    elif data_name == 'heart_disease':
        # https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-3-structured-data-projects/end-to-end-heart-disease-classification.ipynb
        df = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
        # df = pd.read_csv("../data/heart-disease.csv") # Read from local directory, 'DataFrame' shortened to 'df'
        print(df.shape)  # (rows, columns)
        # Everything except target variable
        x_train = df.drop(labels="target", axis=1)
        x_train = x_train.to_numpy()
        x_train = PCA(n_components=6).fit_transform(x_train)
        # Target variable
        y_train = df.target.to_numpy()
        y_train = [[i] for i in y_train]
        num_classes = 3
    elif data_name == 'MedNIST':
        dataDir = 'raw_data/MedNIST_resized'  # The main data directory
        classNames = os.listdir(dataDir)  # Each type of image can be found in its own subdirectory
        numClass = len(classNames)  # Number of types = number of subdirectories
        imageFiles = [
            [os.path.join(dataDir, classNames[i], x) for x in os.listdir(os.path.join(dataDir, classNames[i]))][:20]
            for i in range(numClass)]  # A nested list of filenames
        numEach = [len(imageFiles[i]) for i in range(numClass)]  # A count of each type of image
        imageFilesList = []  # Created an un-nested list of filenames
        imageClass = []  # The labels -- the type of each individual image in the list
        for i in range(numClass):
            imageFilesList.extend(imageFiles[i])
            imageClass.extend([i] * numEach[i])
        numTotal = len(imageClass)  # Total number of images
        imageWidth, imageHeight = Image.open(imageFilesList[0]).size  # The dimensions of each image

        print("There are", numTotal, "images in", numClass, "distinct categories")
        print("Label names:", classNames)
        print("Label counts:", numEach)
        print("Image dimensions:", imageWidth, "x", imageHeight)
        num_classes = len(classNames)  # 6
        x_train = np.asarray([scaleImage(cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in imageFilesList])
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_train = PCA(n_components=6).fit_transform(x_train)
        y_train = imageClass
        # ic(imageClass)
        y_train = [[i] for i in y_train]
        x_train, y_train = shuffle(x_train, y_train, random_state=0)

    elif data_name == 'iris':
        num_classes = 3
        iris = datasets.load_iris()
        x_train = iris.data
        y_train = [[i] for i in iris.target]
        print(x_train, y_train)

    # ic(label_length)
    # Get unique elements and their counts
    # unique_elements, counts = np.unique(label_length, return_counts=True)
    #
    # # Print out the unique elements and their counts
    # for element, count in zip(unique_elements, counts):
    #     print(f"{element} actions label appears {count} times")
    # plt.hist(label_length)
    # plt.xlabel('# actions')
    # plt.ylabel('sample count')
    # plt.show()
    # Generate true weights (theta) for actions
    # true_weights = np.random.randn(n_actions, n_features)
    # norm = np.linalg.norm(true_weights)
    # if norm > 1:
    #     true_weights = true_weights / norm
    #
    # # Initialize context generator
    # generator = ContextGenerator(true_weights=true_weights, noise_std=noise_std)
    # # ic(true_weights.shape)  # true_weights.shape: (4, 4)
    # # Store data for each round
    # data = {
    #     "true_theta": true_weights,
    #     "rounds": []
    # }
    #
    # # Generate data for T rounds
    # for t in range(T):
    #     context, true_rewards = generator.generate_context_and_rewards()
    #
    #     # Store context, true_rewards, and expert_action for each round
    #     data["rounds"].append({
    #         "context": context,  # context.shape: (1, 4)
    #         "true_rewards": true_rewards,  # true_rewards.shape: (4,)
    #     })

    data = {
        "true_theta": [],
        "rounds": []
    }
    label_length = list()
    for t in range(len(x_train)):
        context = np.asarray(x_train[t])[None]  # (1, 294)
        action = np.asarray(y_train[t])  #
        # true_rewards = np.ones_like(action)  #
        true_rewards = np.zeros(num_classes) #
        for i, a in enumerate(action):
            true_rewards[int(a)] = 1.
        if t == 0:
            ic(context.shape, action.shape, true_rewards.shape)
            ic(context, action, true_rewards)
        #     context.shape: (294,)
        #     action.shape: (1,)
        #     true_rewards.shape: (6,)
        label_length.append(len(action))
        data["rounds"].append({
            "context": context,
            "true_rewards": true_rewards,
        })
    print('{} data samples.'.format(len(x_train)))

    return data

if __name__ == "__main__":
    # Argument parser for setting T, n_actions, n_features, and other parameters
    parser = argparse.ArgumentParser(description='Generate Data for T rounds and store in a pickle file')
    parser.add_argument('--T', type=int, default=1000, help='Number of rounds to generate')
    parser.add_argument('--n_actions', type=int, default=4, help='Number of actions')
    parser.add_argument('--n_features', type=int, default=4, help='Number of features for each context')
    parser.add_argument('--noise_std', type=float, default=0, help='Noise standard deviation for reward generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='multilabel_data.pkl', help='Output pickle file to store the data')
    parser.add_argument('--data_name', type=str, default='MedNIST',
                        help='specific data name')

    args = parser.parse_args()

    args.output_file = args.output_file[:-4] + '_{}_{:02d}'.format(args.data_name, args.seed) + args.output_file[-4:]

    # Generate the data
    data = generate_data(T=args.T, n_actions=args.n_actions, n_features=args.n_features, noise_std=args.noise_std, data_name=args.data_name, seed=args.seed)

    # Save data to a pickle file
    with open(args.output_file, 'wb') as f:
        pickle.dump(data, f)
    
    # print(f"Data for {args.T} rounds generated and saved to {args.output_file} with seed {args.seed}")
