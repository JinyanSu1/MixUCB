## Custom version of generate_all_data.py,
## but for the SPANet dataset.

import numpy as np
import pickle
import argparse

from cb_with_human_query.src.utils_contextual_query_food import LinearCBHumanQueryFood
from cb_with_human_query.src.cb_human_query_utils import full_fooditem_list
from sklearn.decomposition import PCA
import pickle as pkl

def load_dataset():
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


def generate_data(T, pca_dim, seed):
    """
    Generates data for T rounds with n_actions, n_features, and noise_std.

    Returns a dictionary with the following structure: (NOTE: doesn't have 'true_theta')
    {
        "rounds": [
            {
                "context": context,
                "true_rewards": true_rewards,
            },
            ...
        ]
    }

    where len(rounds) = T, and context and true_rewards are generated for each round.
    (context is the context at round t, and true_rewards are the true rewards for each action at round t)

    In our case:
    - ignore noise_std.
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
    true_rewards_list = [get_all_dataset_rewards(full_dataset, foodtype, rotationally_symmetric) for foodtype in subsampled_dataset["fooditem"]]

    pca = PCA(n_components=pca_dim)   # run PCA on full dataset.
    contexts_pca = pca.fit_transform(contexts)

    # Generate data for T rounds
    for t in range(T):
        context = contexts_pca[t]
        true_rewards = true_rewards_list[t]
        
        # Store context, true_rewards, and expert_action for each round
        data["rounds"].append({
            "context": context,
            "true_rewards": true_rewards,
        })

    return data

if __name__ == "__main__":
    # Argument parser for setting T, n_actions, n_features, and other parameters
    parser = argparse.ArgumentParser(description='Generate Data for T rounds and store in a pickle file')
    parser.add_argument('--T', type=int, default=1000, help='Number of rounds to generate')
    parser.add_argument('--pca_dim', type=int, default=5, help='PCA dimension to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='simulation_data_spanet.pkl', help='Output pickle file to store the data')
    
    args = parser.parse_args()

    # Generate the data
    data = generate_data(T=args.T, pca_dim=args.pca_dim, seed=args.seed)

    # Save data to a pickle file
    with open(args.output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data for {args.T} rounds generated and saved to {args.output_file} with seed {args.seed}")
