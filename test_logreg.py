from utils.linucb import OnlineLogisticRegressionOracle
import numpy as np
from tqdm import tqdm

def generate_multiclass_data(num_samples=200, num_features=5, num_classes=3, random_seed=42,
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

    return X, y, true_params.T, corrected_y


real_data = False
if real_data:
    # data_name=MedNIST  # heart_disease
    data_name=heart_disease
    PICKLE_FILE="multilabel_data_${data_name}_42.pkl"

    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)

    context = data["rounds"][i]["context"]
    true_rewards = data["rounds"][i]["true_rewards"]
    # TODO define X, y
else:
    n_features = 5
    n_actions = 3
    X, y, true_params, corrected_y = generate_multiclass_data(num_features=n_features, num_classes=n_actions, random_seed=42,
                             temp=1)
    print('linear oracle accuracy', np.sum(corrected_y == y)/len(y))
    normalized_true_params = true_params / np.linalg.norm(true_params, axis=1)[:, np.newaxis]

logreg = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate=0.1, lambda_=1.0, beta=1.0,
                                        mode="torch")

hindsight_acc = []
hindsight_prederr = []
param_errors = []
for feature, label in tqdm(zip(X, y)):
    logreg.update(feature, label)
    theta_hat = logreg.get_model_params()

    normalized_theta_hat = theta_hat / np.linalg.norm(theta_hat, axis=1)[:, np.newaxis]
    param_errors.append(np.linalg.norm(normalized_theta_hat-normalized_true_params))
    hindsight_prederr.append(np.linalg.norm((normalized_theta_hat-normalized_true_params).dot(X.T))**2/X.shape[0])
    preds = logreg.predict(X)
    hindsight_acc.append(np.sum(preds == corrected_y)/len(y))

print(true_params)
print(theta_hat)
import matplotlib.pyplot as plt
plt.figure()
# plt.plot(hindsight_losses, 'losses')
plt.subplot(1,3,1)
plt.plot(param_errors, label='param errors')
plt.subplot(1,3,2)
plt.plot(hindsight_acc, label='acc')
plt.subplot(1,3,3)
plt.plot(hindsight_prederr, label='lin pred errors')
plt.savefig('Figures/logregtest.png')