from utils.linucb import CombinedLinearModel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_multiclass_data(num_samples=2000, num_features=5, num_classes=3, random_seed=42,
                             temp=1):
    """
    Generate a dataset for testing multiclass logistic regression.

    Parameters:
    num_samples (int): Number of samples to generate.
    num_features (int): Number of features.
    num_classes (int): Number of classes.
    random_seed (int): Seed for random number generator.

    Returns:
    pd.DataFrame: DataFrame containing the generated features and class labels.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Generate random feature data
    X = np.random.randn(num_samples, num_features)

    # Generate true linear parameters
    true_params = np.random.randn(num_features, num_classes)

    # Calculate linear combination
    linear_combination = np.dot(X, true_params)

    # Apply softmax to get probabilities
    probabilities = np.exp(temp*linear_combination) / np.sum(np.exp(temp*linear_combination), axis=1, keepdims=True)

    # Generate class labels based on multinomial distribution
    y = np.array([np.argmax(np.random.multinomial(1, pvals)) for pvals in probabilities])

    corrected_y = np.argmax(X.dot(true_params), axis=1)

    return X, y, true_params.T, corrected_y, linear_combination + np.random.normal(scale=0.01, size=linear_combination.shape)


num_samples = 1000
n_features = 5
n_actions = 3
X, y, true_params, corrected_y, realvalues = generate_multiclass_data(num_samples=num_samples, num_features=n_features, num_classes=n_actions, random_seed=42,
                         temp=1)
print('linear oracle accuracy', np.sum(corrected_y == y)/len(y))
normalized_true_params = true_params / np.linalg.norm(true_params, axis=1)[:, np.newaxis]


def compare_delta(theta1, theta2,X=None):
    assert theta1.shape == theta2.shape
    delta1 = theta1 - theta1[0]
    delta2 = theta2 - theta2[0]
    if X is None:
        return np.linalg.norm(delta1-delta2)
    else:
        assert X.shape[1] == theta1.shape[1]
        return np.linalg.norm((delta1-delta2).dot(X.T))**2/X.shape[0]

def compare_param(theta1, theta2,X=None):
    assert theta1.shape == theta2.shape
    delta1 = theta1
    delta2 = theta2
    if X is None:
        return np.linalg.norm(delta1-delta2)
    else:
        assert X.shape[1] == theta1.shape[1]
        return np.linalg.norm((delta1-delta2).dot(X.T))**2/X.shape[0]


nums_log_data = np.linspace(100, num_samples, 10)
percents_lin_data = [0, 2, 5, 10, 25, 100] #np.linspace(2,100,5)
performance = {'param':np.zeros((len(nums_log_data),len(percents_lin_data))), 
               'acc':np.zeros((len(nums_log_data),len(percents_lin_data))), 
                'corrected_acc':np.zeros((len(nums_log_data),len(percents_lin_data))), 
                'prediction':np.zeros((len(nums_log_data),len(percents_lin_data)))}
for i,n_ld in enumerate(nums_log_data):
    for j,p_ld in enumerate(percents_lin_data):
        model = CombinedLinearModel(n_features,n_actions,lr=0.1, weight_decay=0.01)
        lr_ind = int(np.ceil(p_ld*n_ld/100))
        if p_ld > 0:
            model.fit(X[:int(n_ld)], y[:int(n_ld)], X[:lr_ind], realvalues[:lr_ind])
        else:
            model.fit(X[:int(n_ld)], y[:int(n_ld)])
        theta_hat = model.coef_
        preds_y = model.predict(X)

        performance['acc'][i,j] = np.sum(preds_y == y)/len(y)
        performance['param'][i,j] = compare_param(theta_hat,true_params)
        performance['corrected_acc'][i,j] = np.sum(preds_y == corrected_y)/len(y)
        performance['prediction'][i,j] = compare_param(theta_hat,true_params, X)

fig, axs = plt.subplots(1, len(performance.keys()), figsize=(12, 4))

for j,key in enumerate(performance.keys()):
    for i, p_ld in enumerate(percents_lin_data):
        axs[j].semilogy(nums_log_data, performance[key][:,i],label=str(p_ld))
    axs[j].set_title(key)
    axs[j].set_xlabel('n logreg samples')
    axs[j].legend(title='% linreg sample')

fig.tight_layout()
fig.savefig('Figures/logregtest.png')
