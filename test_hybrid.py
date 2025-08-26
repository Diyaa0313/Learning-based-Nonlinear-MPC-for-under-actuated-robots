import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from dynamics_learner.gp import SlidingWindowGP
from dynamics_learner.nn import ResidualNN


# -------------------- Data utils --------------------
def load_pendubot(path="pendubot_residuals.npy"):
    """
    Expects npy saved as an array/list of (x, u, r) tuples.
    x: [x_dim], u: [u_dim], r: [r_dim]
    Returns:
      inputs = [N, x_dim+u_dim], targets = [N, r_dim]
    """
    data = np.load(path, allow_pickle=True)
    X, U, R = [], [], []
    for (x, u, r) in data:
        X.append(np.asarray(x, dtype=float).ravel())
        U.append(np.asarray(u, dtype=float).ravel())
        R.append(np.asarray(r, dtype=float).ravel())
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    R = np.asarray(R, dtype=float)
    inputs = np.concatenate([X, U], axis=1)
    targets = R
    return inputs, targets


def chronological_split(X, Y, train_ratio=0.7):
    N = len(X)
    ntr = int(train_ratio * N)
    return X[:ntr], X[ntr:], Y[:ntr], Y[ntr:]


def standardize_train_test(train_in, test_in, train_out, test_out):
    in_mean, in_std = train_in.mean(0), train_in.std(0) + 1e-8
    out_mean, out_std = train_out.mean(0), train_out.std(0) + 1e-8

    train_in_n = (train_in - in_mean) / in_std
    test_in_n  = (test_in  - in_mean) / in_std
    train_out_n = (train_out - out_mean) / out_std
    test_out_n  = (test_out  - out_mean) / out_std

    stats = {"in_mean": in_mean, "in_std": in_std, "out_mean": out_mean, "out_std": out_std}
    return train_in_n, test_in_n, train_out_n, test_out_n, stats


def denorm(y, mean, std):
    return y * std + mean


def rolling_metric(y_true, y_pred, window=30, metric="mse"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    T = y_true.shape[0]
    out = np.zeros(T)
    for t in range(T):
        s = max(0, t - window + 1)
        yt = y_true[s:t+1]
        yp = y_pred[s:t+1]
        if metric == "mse":
            err = np.mean((yp - yt) ** 2)
        else:
            err = np.mean(np.abs(yp - yt))
        out[t] = err
    return out


# -------------------- Hybrid (uncertainty-weighted) --------------------
# def fuse_uncertainty(mean_gp, var_gp, mean_nn, var_nn, eps=1e-6):
#     """
#     Per-dimension inverse-variance weighting.
#     mean = (m_g/σ_g^2 + m_n/σ_n^2) / (1/σ_g^2 + 1/σ_n^2)
#     var  = 1 / (1/σ_g^2 + 1/σ_n^2)
#     """
#     w_gp = 1.0 / (np.asarray(var_gp) + eps)
#     w_nn = 1.0 / (np.asarray(var_nn) + eps)
#     wsum = w_gp + w_nn
#     mean = (w_gp * mean_gp + w_nn * mean_nn) / wsum
#     var = 1.0 / wsum
#     return mean, var


# -------------------- Main --------------------
def main():
    # ---------- Load & split ----------
    inputs, targets = load_pendubot("pendubot_residuals.npy")
    x_dim = inputs.shape[1] - 1  # (pendubot: 4 states + 1 input)
    u_dim = 1
    r_dim = targets.shape[1]

    train_in, test_in, train_out, test_out = chronological_split(inputs, targets, train_ratio=0.7)
    (train_in_n, test_in_n, train_out_n, test_out_n, stats) = standardize_train_test(
        train_in, test_in, train_out, test_out
    )
    out_mean, out_std = stats["out_mean"], stats["out_std"]

    # ---------- Instantiate learners ----------
    gp = SlidingWindowGP(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim,
                         maxlen=300, lengthscale=1.0, variance=1.0, noise_variance=1e-3)

    nn = ResidualNN(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim,
                    hidden=(256, 128, 64), dropout=0.15, lr=3e-4, weight_decay=1e-4)

    # ---------- Train (chronological) ----------
    for i in range(len(train_in_n)):
        x = train_in_n[i, :x_dim]
        u = train_in_n[i, x_dim:]
        r = train_out_n[i]
        gp.update([(x, u, r)])

    nn_batch = [(train_in_n[i, :x_dim], train_in_n[i, x_dim:], train_out_n[i])
                for i in range(len(train_in_n))]
    nn.update(nn_batch, epochs=120, patience=15, batch_size=64)

    # ---------- Evaluate on test ----------
    def run_model_means_and_vars(model):
        means, vars_ = [], []
        for i in range(len(test_in_n)):
            x = test_in_n[i, :x_dim]
            u = test_in_n[i, x_dim:]
            m, v = model.predict(x, u)
            means.append(m)
            vars_.append(v)
        return np.asarray(means), np.asarray(vars_)

    gp_means_n, gp_vars_n = run_model_means_and_vars(gp)
    nn_means_n, nn_vars_n = run_model_means_and_vars(nn)

    # hy_means_n, hy_vars_n = [], []
    # for i in range(len(test_in_n)):
    #     m_f, v_f = fuse_uncertainty(gp_means_n[i], gp_vars_n[i], nn_means_n[i], nn_vars_n[i])
    #     hy_means_n.append(m_f); hy_vars_n.append(v_f)
    # hy_means_n = np.asarray(hy_means_n)

    gp_means = denorm(gp_means_n, out_mean, out_std)
    nn_means = denorm(nn_means_n, out_mean, out_std)
    # hy_means = denorm(hy_means_n, out_mean, out_std)
    truths   = denorm(test_out_n, out_mean, out_std)

    # ---------- Metrics ----------
    def summary(name, pred):
        mse = float(np.mean((pred - truths) ** 2))
        mae = float(np.mean(np.abs(pred - truths)))
        print(f"{name}: MSE={mse:.6f}, MAE={mae:.6f}")

    print("==== Test (de-normalized) ====")
    summary("GP", gp_means)
    summary("NN", nn_means)
    # summary("Hybrid (uncertainty-weighted)", hy_means)

    # ---------- Rolling curves ----------
    T = len(truths)
    t = np.arange(T)
    roll_w = max(25, T // 6)
    gp_mse = rolling_metric(truths, gp_means, roll_w, "mse")
    nn_mse = rolling_metric(truths, nn_means, roll_w, "mse")
    # hy_mse = rolling_metric(truths, hy_means, roll_w, "mse")
    gp_mae = rolling_metric(truths, gp_means, roll_w, "mae")
    nn_mae = rolling_metric(truths, nn_means, roll_w, "mae")
    # hy_mae = rolling_metric(truths, hy_means, roll_w, "mae")

    # ---------- Plots ----------
    # Residual tracking only for dim 0
    # ---------- Save results to PDF ----------
    with PdfPages("results.pdf") as pdf:
        # Residual tracking only for dim 0
        plt.figure(figsize=(12, 4))
        plt.plot(t, truths[:, 0], label="True residual (dim 0)")
        plt.plot(t, gp_means[:, 0], label="GP")
        plt.plot(t, nn_means[:, 0], label="NN")
        plt.title("Residual tracking (dim 0)")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Rolling MSE
        plt.figure(figsize=(12, 4))
        plt.plot(t, gp_mse, label="GP")
        plt.plot(t, nn_mse, label="NN")
        plt.title("Rolling MSE (de-normalized)")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Rolling MAE
        plt.figure(figsize=(12, 4))
        plt.plot(t, gp_mae, label="GP")
        plt.plot(t, nn_mae, label="NN")
        plt.title("Rolling MAE (de-normalized)")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Add a summary text page
        fig = Figure(figsize=(8.5, 6))
        ax = fig.add_subplot(111)
        ax.axis("off")
        text = (
            "==== Test Results (de-normalized) ====\n"
            f"GP:  MSE={np.mean((gp_means - truths)**2):.6f}, "
            f"MAE={np.mean(np.abs(gp_means - truths)):.6f}\n"
            f"NN:  MSE={np.mean((nn_means - truths)**2):.6f}, "
            f"MAE={np.mean(np.abs(nn_means - truths)):.6f}\n"
        )
        ax.text(0.05, 0.95, text, va="top", ha="left", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(" Results saved to results.pdf")



if __name__ == "__main__":
    main()
