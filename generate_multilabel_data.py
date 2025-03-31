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
from sklearn.preprocessing import normalize
from utils.regression_ucb import CombinedLinearModel
from pathlib import Path

# SPANET imports
from cb_with_human_query.src.utils_contextual_query_food import LinearCBHumanQueryFood
from cb_with_human_query.src.cb_human_query_utils import full_fooditem_list
import pickle as pkl
## UTILITIES

def hindsight_theta(X, y, n_features, n_actions,  mode="regression"):
    assert mode in ["regression", "classification"]
    model = CombinedLinearModel(n_features, n_actions, lr=0.1, weight_decay=0)
    if mode == "regression":
        model.fit([],[],X_sq=X,y_sq=y)
    else:
        model.fit(X, y)
    return model.coef_

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

## UTILITIES, FOR SPANET DATA.

def load_dataset():
    """
    Returns:
        - food_dataset: full dataset of (context, action, reward) pairs.
        - pretrain_dataset: dataset of (context, action, reward) pairs for pretraining.
        - val_dataset: dataset of (context, action, reward) pairs for validation.
        - rotationally_symmetric: set of food items that are rotationally symmetric.
    """
    validation_fooditems = []
    pretrain_fooditems = list(set(full_fooditem_list) - set(validation_fooditems))

    # Dig up dataset of (context, action, reward) pairs
    # copied from utils_contextual_query_food.
    path_to_food_dataset = "cb_with_human_query/feeding_preprocessing/spanet_dataset_with_contexts_iros2024_carrot_banana_cantaloupe_grape.pkl"

    with open(path_to_food_dataset,"rb") as f:
        food_dataset = pkl.load(f)
    print(f"Loading food dataset located at {path_to_food_dataset}")
    # Create new action column that maps actions to integers.
    food_dataset["action"] = food_dataset.apply(LinearCBHumanQueryFood.convert_action_to_int, axis=1)
   
    # Remove rows with null contexts.
    food_dataset = food_dataset[~food_dataset.isna()["context"]]

    # Extract set of rotationally-symmetric food-items (used for ground-truth reward computation)
    rotationally_symmetric = []
    for fooditem in food_dataset["fooditem"].unique():
        missing_90_data = False
        for action in ["tilted_angled","tilted_vertical_skewer","vertical_skewer"]:
            for roll in ['0','90']:
                base_df = food_dataset
                df = base_df[(base_df['fooditem']==fooditem) & (base_df['action_pitch']==action) & (base_df['action_roll']==roll)]
                if roll == '90' and np.isnan(np.mean(df['success'])):
                    missing_90_data = True
        if missing_90_data:
            rotationally_symmetric.append(fooditem)
    print(f"Rotationally symmetric food items: {rotationally_symmetric}")

    # Filter validation food items if necessary.
    pretrain_dataset = food_dataset[food_dataset["fooditem"].isin(pretrain_fooditems)]
    val_dataset = food_dataset[food_dataset["fooditem"].isin(validation_fooditems)]
    print(f"Pretrain food items: {pretrain_fooditems}")
    print(f"Validation food items: {validation_fooditems}, Full dataset size: {len(food_dataset)}")

    return food_dataset, pretrain_dataset, val_dataset, rotationally_symmetric


def dataset_reward(food_dataset, foodtype, arm, rotationally_symmetric):
    """
    Return mean reward for a given foodtype and arm.
    """
    # If the foodtype is rotationally symmetric, then we will lookup the mapping for the action with 0-degree roll.
    arm_to_use = arm
    if foodtype in rotationally_symmetric:
        pitch, _ = LinearCBHumanQueryFood.convert_int_to_action(arm)
        arm_to_use = LinearCBHumanQueryFood.convert_action_to_int({"action_pitch":pitch, "action_roll":'0'})

    # Compute the mean reward for this arm and context, using food_dataset.

    # Pick rows corresponding to action i, that have non-null context.
    action_food_dataset = food_dataset[(food_dataset["action"]==arm_to_use) & (~food_dataset.isna()["context"])]
    # Pick rows corresponding to the given foodtype.
    action_food_dataset = action_food_dataset[action_food_dataset["fooditem"]==foodtype]
    # Compute mean reward.
    mean_reward = action_food_dataset["success"].mean()
    return mean_reward

def get_all_dataset_rewards(food_dataset, foodtype, rotationally_symmetric):
    """
    Return all rewards for a given foodtype.
    """
    reward_list = []
    num_arms = 6
    for arm in range(num_arms):
        mean_reward = dataset_reward(food_dataset, foodtype, arm, rotationally_symmetric)
        reward_list.append(mean_reward)

    return reward_list

## MAIN METHODS.

def generate_synthetic_data(T, noise_std, seed):
    """
    Seed should affect the following: context sequence, true rewards, (indirectly - expected rewards), expert choices.
    Derived from: https://github.com/JinyanSu1/MixUCB/blob/devel/rohan/generate_toy_data.py

    Returns:
        data (fields for data: true_theta, true_theta_classification, rounds;
        fields for data['rounds'] values: context, actual_rewards, expected_rewards, noisy_expert_choice)
    """
    # Set np global random seed for reproducibility
    np.random.seed(seed)

    # Generate true weights (theta) for actions
    true_weights = np.array([[np.cos(0),np.sin(0)],[np.cos(2/3*np.pi),np.sin(2/3*np.pi)],[np.cos(4/3*np.pi),np.sin(4/3*np.pi)]])

    # Initialize context generator
    generator = ContextGenerator(true_weights=true_weights, noise_std=noise_std)

    # Store data for each round
    data = {
        "true_theta": true_weights,
        "true_theta_classification": true_weights,
        "rounds": []
    }

    # Generate data for T rounds
    for t in range(T):
        context, noisy_rewards, noiseless_rewards, noisy_expert_choice = generator.generate_context_and_rewards()

        # Normalize the context
        context_normalized = normalize(context)
        
        # Store context, true_rewards, and expert_action for each round
        data["rounds"].append({
            "context": context_normalized,
            "actual_rewards": noisy_rewards,                       # actual/observed rewards - should be noisy. use for evaluation.
            "expected_rewards": noiseless_rewards,                 # expected rewards - should be noiseless (otherwise, expert is anticipating noise). 
                                                                   # use for expert decision-making
            "noisy_expert_choice": noisy_expert_choice             # boltzmann expert choice that operates on expected rewards.
        })

    return data

def generate_spanet_data(T, pca_dim, seed):
    """
    Generates data for T rounds.
    Seed should affect the following: context sequence, true rewards, (indirectly - expected rewards), expert choices.

    Derived from: https://github.com/JinyanSu1/MixUCB/blob/devel/rohan/generate_spanet_data.py

    Returns:
        data (fields for data: true_theta, true_theta_classification, rounds;
        fields for data['rounds'] values: context, actual_rewards, expected_rewards, noisy_expert_choice)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Store data for each round
    data = {
        "rounds": []
    }

    full_dataset, _, _, rotationally_symmetric = load_dataset()
    # Sample T random contexts and true rewards for each round.
    subsampled_dataset = full_dataset.sample(n=T, random_state=seed)
    contexts = np.squeeze(np.array(list(subsampled_dataset["context"])))

    # Oracle is the dataset itself.
    # Generates list of expected rewards.
    true_rewards_list = [get_all_dataset_rewards(full_dataset, foodtype, rotationally_symmetric) for foodtype in subsampled_dataset["fooditem"]]
    
    pca_full = PCA(n_components=pca_dim)
    pca_full.fit(np.squeeze(np.array(list(full_dataset["context"]))))

    # PCA just for the contexts in the dataset.
    contexts_pca = pca_full.fit_transform(contexts)

    # Normalize the pca_contexts --> x_train; true_rewards_list --> y_train.
    contexts_pca_normalized = normalize(contexts_pca)
    x_train = contexts_pca_normalized
    y_train = np.zeros((len(contexts_pca_normalized), len(true_rewards_list[0])))
    for i, true_rewards in enumerate(true_rewards_list):
        y_train[i] = true_rewards
    # Add true theta values to data dictionary.
    true_theta = hindsight_theta(np.array(x_train),np.vstack(y_train),len(x_train[0]), len(y_train[0]))
    true_theta_classification = hindsight_theta(np.array(x_train),np.argmax(y_train, axis=1),len(x_train[0]), len(y_train[0]), mode="classification")
    data["true_theta"] = true_theta
    data["true_theta_classification"] = true_theta_classification

    # Generate data for T rounds
    for t in range(T):
        context = contexts_pca_normalized[t]
        true_rewards = true_rewards_list[t]
        # Generate noisy expert choice based on expected (noiseless) rewards.
        r=1 # rationality.
        noisy_expert_choice = np.random.choice(len(true_rewards), p=np.exp(r*true_rewards)/sum(np.exp(r*true_rewards)))

        # Store context, actual_rewards, and expected_rewards for each round
        data["rounds"].append({
            "context": np.expand_dims(context,0),
            "actual_rewards": np.random.binomial(1, p=true_rewards),
            "expected_rewards": true_rewards,
            "noisy_expert_choice": noisy_expert_choice
        })

    return data

def generate_medical_data(T, seed, data_name='heart_disease', norm_features=False):
    '''
    Specifically for medical datasets.

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
        # shuffle data
        x_train, y_train = shuffle(x_train, y_train, random_state=seed)
        # truncate at T
        x_train = x_train[:T]
        y_train = y_train[:T]
    elif data_name == 'heart_disease':  # not shuffle
        '''
        # context.shape: (1, 6)                                                                                                                                                                                                                           
        # action.shape: (1,)                                                                                                                                                                                                                              
        # true_rewards.shape: (3,) 
        '''
        # https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-3-structured-data-projects/end-to-end-heart-disease-classification.ipynb
        df = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
        # df = pd.read_csv("../data/heart-disease.csv") # Read from local directory, 'DataFrame' shortened to 'df'
        print(df.shape)  # (rows, columns)
        # Everything except target variable
        x_train = df.drop(labels="target", axis=1)
        x_train = x_train.to_numpy()
        x_train = PCA(n_components=6).fit_transform(x_train)
        x_train = x_train[::-1]
        # Target variable
        y_train = df.target.to_numpy()
        y_train = [[i] for i in y_train[::-1]]
        num_classes = 2
        # shuffle data
        x_train, y_train = shuffle(x_train, y_train, random_state=seed)
        # truncate at T
        x_train = x_train[:T]
        y_train = y_train[:T]

    elif data_name == 'MedNIST':
        '''
        # There are 120 images in 6 distinct categories                                                                                                                                                                                                       
        # Label names: ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']                                                                                                                                                                         
        # Label counts: [20, 20, 20, 20, 20, 20]                                                                                                                                                                                                              
        # Image dimensions: 64 x 64                                                                                                                                                                                                                           
        # context.shape: (1, 6)                                                                                                                                                                                                                           
        # action.shape: (1,)                                                                                                                                                                                                                              
        # true_rewards.shape: (6,) 
        '''
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
        # shuffle data according to seed
        x_train, y_train = shuffle(x_train, y_train, random_state=seed)
        # truncate at T
        x_train = x_train[:T]
        y_train = y_train[:T]

    elif data_name == 'iris':
        num_classes = 3
        iris = datasets.load_iris()
        x_train = iris.data
        y_train = [[i] for i in iris.target]
        # shuffle data
        x_train, y_train = shuffle(x_train, y_train, random_state=seed)
        # truncate at T
        x_train = x_train[:T]
        y_train = y_train[:T]
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

    if norm_features:
        x_train = normalize(x_train)

    onehotlabels = np.zeros((len(x_train), num_classes))
    onehotlabels[y_train] = 1
    true_theta = hindsight_theta(np.array(x_train),onehotlabels,len(x_train[0]), num_classes)
    true_theta_classification = hindsight_theta(np.array(x_train),np.array(y_train).ravel(),len(x_train[0]), num_classes, mode="classification")
    
    data = {
        "true_theta": true_theta,
        "true_theta_classification": true_theta_classification,
        "rounds": []
    }

    for t in range(len(x_train)):
        context = np.asarray(x_train[t])[None]  # (1, 294)
        action = np.asarray(y_train[t])  #
        actual_rewards = np.zeros(num_classes) #
        for i, a in enumerate(action):
            actual_rewards[int(a)] = 1.
        # TODO true reward vs expected reward

        data["rounds"].append({
            "context": context,
            "actual_rewards": actual_rewards,
            "expected_rewards": np.array([]),
            "noisy_expert_choice": np.array([])
        })
    print('{} data samples.'.format(len(x_train)))

    return data

def reprocess_pkl(filename):
    """
    Reprocesses an existing data pickle file. Handles either synthetic data or non-synthetic data (spanet data).
    - Normalizes contexts.
    - Calculates (if applicale) and adds "true_theta" and "true_theta_classification" to data dictionary.
    - Populates 'actual_rewards' and 'expected_rewards' for each round.
    
    Dumps the reprocessed data to a new pickle file with the same name as the input file, but with 'reprocessed' appended to the name.
    """
    # Compile x and y labels. Also normalizes x.
    # x --> context
    # y --> true_rewards (synthetic: these are actual rewards, non-synthetic: these are expected rewards)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    x_train = []
    y_train = []
    for step in data["rounds"]:
        x_train.append(step["context"].ravel().astype(np.float64))
        y_train.append(np.array(step["true_rewards"]).ravel().astype(np.float64))
    x_train = normalize(x_train)

    synthetic = False

    if "true_theta" in data.keys():
        # synethic case, generating theta
        synthetic = True
        data["true_theta_classification"] = data["true_theta"]
    else:
        # non-synthetic case. Generating thetas from hindsight (both regression and classification)
        true_theta = hindsight_theta(np.array(x_train),np.vstack(y_train),len(x_train[0]), len(y_train[0]))
        true_theta_classification = hindsight_theta(np.array(x_train),np.argmax(y_train, axis=1),len(x_train[0]), len(y_train[0]), mode="classification")
        data["true_theta"] = true_theta
        data["true_theta_classification"] = true_theta_classification

    # Populate data with new context and new expected rewards.
    for t in range(len(data["rounds"])):
        data["rounds"][t]["context"] = x_train[t].reshape(1,-1)
        if synthetic:
            data["rounds"][t]["actual_rewards"] = data["rounds"][t]["true_rewards"]
            data["rounds"][t]["expected_rewards"] = np.dot(data["true_theta"],x_train[t])
        else:
            # sample 0/1 success
            data["rounds"][t]["actual_rewards"] = np.random.binomial(1, p=data["rounds"][t]["true_rewards"])
            data["rounds"][t]["expected_rewards"] = data["rounds"][t]["true_rewards"]
    
    newfile = filename[:-4] + 'reprocessed.pkl'
    with open(newfile, 'wb') as f:
        pickle.dump(data, f)


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
    parser.add_argument('--reprocess', type=str, default='')

    args = parser.parse_args()

    if len(args.reprocess) > 0:
        reprocess_pkl(args.reprocess) 
    else:
        raw_data_dir = Path("raw_data")

        args.output_file = args.output_file[:-4] + '_{}_{:02d}'.format(args.data_name, args.seed) + args.output_file[-4:]

        # Generate the data
        if args.data_name == 'synthetic':
            data = generate_synthetic_data(T=args.T, noise_std=args.noise_std, seed=args.seed)
        elif args.data_name == 'spanet':
            data = generate_spanet_data(T=args.T, pca_dim=args.n_features, seed=args.seed)
        elif args.data_name == 'heart_disease' or args.data_name == 'MedNIST' or args.data_name == 'yeast' or args.data_name == 'iris':
            data = generate_medical_data(T=args.T, data_name=args.data_name, seed=args.seed, norm_features=True)

        # Save data to a pickle file
        with open(Path(raw_data_dir / args.output_file), 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Data for {args.T} rounds generated and saved to {args.output_file} with seed {args.seed}")
