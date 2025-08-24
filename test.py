# test.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dynamics_learner.gp import SlidingWindowGP
from dynamics_learner.nn import ResidualNN

# ------------------ Load dataset ------------------
dataset = np.load("pendubot_residuals.npy", allow_pickle=True)
print(f"Loaded dataset with {len(dataset)} samples")

X, U, R = [], [], []
for (x, u, r) in dataset:
    X.append(x)
    U.append(u)
    R.append(r)
X = np.array(X)
U = np.array(U)
R = np.array(R)

# Concatenate state+action for learning
inputs = np.concatenate([X, U], axis=1)  # shape (N, 5)
targets = R                              # shape (N, 4)

# ------------------ Standardize ------------------
in_mean, in_std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-8
out_mean, out_std = targets.mean(axis=0), targets.std(axis=0) + 1e-8

inputs = (inputs - in_mean) / in_std
targets = (targets - out_mean) / out_std

# Train/test split
train_in, test_in, train_out, test_out = train_test_split(
    inputs, targets, test_size=0.2, random_state=42
)

# ------------------ GP ------------------
gp = SlidingWindowGP(x_dim=4, u_dim=1, r_dim=4, maxlen=200)
for i in range(len(train_in)):
    x = train_in[i, :4]
    u = train_in[i, 4:]
    r = train_out[i]
    gp.update([(x, u, r)])

gp_means, gp_vars, gp_truths = [], [], []
for i in range(len(test_in)):
    x = test_in[i, :4]
    u = test_in[i, 4:]
    r = test_out[i]
    mean, var = gp.predict(x, u)
    gp_means.append(mean)
    gp_vars.append(var)
    gp_truths.append(r)

gp_means = np.array(gp_means)
gp_vars = np.array(gp_vars)
gp_truths = np.array(gp_truths)

# ------------------ NN ------------------
nn = ResidualNN(x_dim=4, u_dim=1, r_dim=4, hidden=(128, 128, 64))

batch = [(train_in[i, :4], train_in[i, 4:], train_out[i]) for i in range(len(train_in))]
nn.update(batch, epochs=100, patience=10)

nn_means, nn_vars, nn_truths = [], [], []
for i in range(len(test_in)):
    x = test_in[i, :4]
    u = test_in[i, 4:]
    r = test_out[i]
    mean, var = nn.predict(x, u)
    nn_means.append(mean)
    nn_vars.append(var)
    nn_truths.append(r)

nn_means = np.array(nn_means)
nn_vars = np.array(nn_vars)
nn_truths = np.array(nn_truths)

# ------------------ Train/Test Metrics ------------------
def denorm(y): return y * out_std + out_mean

# Train set predictions
train_preds = []
for i in range(len(train_in)):
    m, _ = nn.predict(train_in[i, :4], train_in[i, 4:])
    train_preds.append(m)
train_preds = np.array(train_preds)

# Test set predictions
test_preds = []
for i in range(len(test_in)):
    m, _ = nn.predict(test_in[i, :4], test_in[i, 4:])
    test_preds.append(m)
test_preds = np.array(test_preds)

print("==== NN Performance (de-normalized residuals) ====")
print("Train MAE:", mean_absolute_error(denorm(train_out), denorm(train_preds)))
print("Train MSE:", mean_squared_error(denorm(train_out), denorm(train_preds)))
print("Test  MAE:", mean_absolute_error(denorm(test_out), denorm(test_preds)))
print("Test  MSE:", mean_squared_error(denorm(test_out), denorm(test_preds)))

# ------------------ De-normalize outputs for plotting ------------------
gp_means = gp_means * out_std + out_mean
gp_truths = gp_truths * out_std + out_mean
nn_means = nn_means * out_std + out_mean
nn_truths = nn_truths * out_std + out_mean

# ------------------ Plotting ------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# GP
axs[0].plot(gp_truths[:, 0], label="True residual (dim 0)", color="black")
axs[0].plot(gp_means[:, 0], label="GP mean", color="blue")
axs[0].fill_between(range(len(gp_vars)),
                    gp_means[:, 0] - np.sqrt(gp_vars[:, 0]),
                    gp_means[:, 0] + np.sqrt(gp_vars[:, 0]),
                    color="blue", alpha=0.2, label="GP ±1 std")
axs[0].set_title("Sliding Window GP (dim 0)")
axs[0].legend()

# NN
axs[1].plot(nn_truths[:, 0], label="True residual (dim 0)", color="black")
axs[1].plot(nn_means[:, 0], label="NN mean", color="red")
axs[1].fill_between(range(len(nn_vars)),
                    nn_means[:, 0] - np.sqrt(nn_vars[:, 0]),
                    nn_means[:, 0] + np.sqrt(nn_vars[:, 0]),
                    color="red", alpha=0.2, label="NN ±1 std")
axs[1].set_title("Residual NN with MC Dropout (dim 0)")
axs[1].legend()

plt.tight_layout()
plt.show()
