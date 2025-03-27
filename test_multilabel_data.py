## Unit tests.

import generate_multilabel_data
import utils.get_data
import numpy as np

def test_generate_synthetic_data():
    """
    Sanity-checks on synthetic data generation.
    """
    T = 1000
    noise_std = 0
    seed = 42
    data = generate_multilabel_data.generate_synthetic_data(T, noise_std, seed)
    # Fixed parameters
    num_actions = 3
    num_features = 2
    # Sanity checks on first data-point.
    assert data["true_theta"].shape == (num_actions, num_features)
    assert len(data["rounds"]) == T
    assert data["rounds"][0]["context"].shape == (1,num_features), f"Context shape: {data['rounds'][0]['context'].shape}"
    assert len(data["rounds"][0]["actual_rewards"]) == num_actions
    assert len(data["rounds"][0]["expected_rewards"]) == num_actions
    assert data["rounds"][0]["noisy_expert_choice"] in range(num_actions)

# Test to make sure that expert choice from ContextGenerator is based on noiseless rewards, not noisy rewards.
def test_context_generator():
    noise_std = 0.2
    seed = 42
    num_actions = 3
    num_features = 2
    np.random.seed(seed)
    context_generator = utils.get_data.ContextGenerator(np.array([[1,0],[0,1],[1,1]]), noise_std=noise_std)
    context, noisy_rewards, noiseless_rewards, noisy_expert_choice = context_generator.generate_context_and_rewards()
    # Softmax action based on noiseless rewards.
    r=1
    np.random.seed(seed)
    _ = np.random.randn(1, num_features)
    _ = np.random.randn(num_actions)                # need these to ensure random seed is in the same state as it was when generating the data.
    noisy_expert_choice_softmax = np.random.choice(len(noiseless_rewards), p=np.exp(r*noiseless_rewards)/sum(np.exp(r*noiseless_rewards)))
    assert noisy_expert_choice == noisy_expert_choice_softmax



if __name__=="__main__":
    test_generate_synthetic_data()
    test_context_generator()