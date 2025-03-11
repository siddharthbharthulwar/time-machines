#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Hyperparameters
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_TOTAL = 20              # total distinct 'epochs' or time indices we want to memorize
BATCH_SIZE = 128               # training batch size
TRAIN_STEPS = 3000             # number of training steps (you can increase this)
LR = 1e-3                      # learning rate
EMBED_DIM = 16                 # dimension for epoch embedding
DIM_X = 2                      # dimension of data
DATA_STD = 1.0                 # std dev of synthetic data

# Diffusion hyperparams
DIFFUSION_STEPS = 100          # number of diffusion steps T
BETA_START = 1e-4
BETA_END   = 2e-2
betas = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS).to(DEVICE)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # \bar{\alpha}_t

# ----------------------------
# 2. Synthetic Dataset
# ----------------------------
# We'll define a simple infinite data loader that samples on the fly:
# - For epoch e in [0..9], x ~ N([-5, 5], I)
# - For epoch e in [10..19], x ~ N([ 5, 5], I)

def sample_data(batch_size):
    # Randomly sample an integer epoch in [0, EPOCHS_TOTAL)
    # Then sample data from the correct distribution.
    epoch_indices = torch.randint(0, EPOCHS_TOTAL, (batch_size,), device=DEVICE)

    # Means for each epoch: first 10 = (-5, 5), second 10 = (5, 5)
    # We'll create a (batch_size, 2) for the means.
    # e < 10 => mean=(-5,5); otherwise => mean=(5,5)
    means = torch.empty((batch_size, DIM_X), device=DEVICE)
    mask = (epoch_indices < 10).float().unsqueeze(-1)  # shape (batch_size, 1)
    means = mask * torch.tensor([-5.0, 5.0], device=DEVICE) \
            + (1 - mask) * torch.tensor([5.0, 5.0], device=DEVICE)

    # Sample from Normal(mean, I)
    x = means + DATA_STD * torch.randn((batch_size, DIM_X), device=DEVICE)
    return x, epoch_indices

# ----------------------------
# 3. Model Definition
# ----------------------------
# We'll define a small MLP to predict the noise in x_t given (x_t, t, epoch).
# The model architecture:
#   - Embedding for epoch index -> embed_dim
#   - Embedding for diffusion step t -> embed_dim (using sinusoidal or learned)
#   - MLP that takes [x_t, epoch_embed, t_embed] -> predicts noise (same dimension as x_t)

class SinusoidalPositionalEmbedding(nn.Module):
    """
    A standard sinusoidal positional embedding for the diffusion time step t.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        t: (batch_size,) with integer or float time steps
        returns: (batch_size, embed_dim)
        """
        # Normalize t to [0,1] for embedding, or we can keep it raw.
        # We'll just keep t in [0, DIFFUSION_STEPS) and do standard sinusoidal.
        half_dim = self.embed_dim // 2
        # create a range
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=t.device).float() / half_dim
        )
        # shape of freqs: (half_dim,)
        # expand t to (batch_size, half_dim)
        t = t.unsqueeze(-1).float()
        # shape -> (batch_size, half_dim)
        angles = t * freqs
        # embed = [sin(angles), cos(angles)]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb

class DiffusionModel(nn.Module):
    def __init__(self, dim_x, embed_dim, hidden_dim=128):
        super().__init__()
        self.dim_x = dim_x
        self.embed_dim = embed_dim

        # Embedding for epoch index
        self.epoch_embedding = nn.Embedding(EPOCHS_TOTAL, embed_dim)

        # Embedding for diffusion step t
        self.t_embedding = SinusoidalPositionalEmbedding(embed_dim)

        # MLP that will take in (x_t, epoch_embed, t_embed) -> noise
        input_dim = dim_x + embed_dim + embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_x),
        )

    def forward(self, x_t, t, epoch_idx):
        # x_t: (batch_size, dim_x)
        # t: (batch_size,) in [0..DIFFUSION_STEPS-1]
        # epoch_idx: (batch_size,) in [0..EPOCHS_TOTAL-1]
        epoch_emb = self.epoch_embedding(epoch_idx)         # (batch_size, embed_dim)
        t_emb = self.t_embedding(t)                         # (batch_size, embed_dim)
        inp = torch.cat([x_t, epoch_emb, t_emb], dim=-1)    # (batch_size, dim_x + 2*embed_dim)
        noise_pred = self.net(inp)
        return noise_pred

# ----------------------------
# 4. Diffusion Utilities
# ----------------------------
# Forward noising process:
#    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) I)
#
# During training, we sample x_0 from data,
# pick a random t in [1..DIFFUSION_STEPS], then generate x_t, and train
# the model to predict the noise that was added.

def q_sample(x_0, t):
    """
    Forward diffuse x_0 to x_t:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    """
    # alpha_bar_t has shape (batch_size,) after gathering
    sqrt_alpha_bar_t = alpha_bars[t].sqrt().unsqueeze(-1)
    sqrt_1_minus_alpha_bar_t = (1 - alpha_bars[t]).sqrt().unsqueeze(-1)

    eps = torch.randn_like(x_0)
    return sqrt_alpha_bar_t * x_0 + sqrt_1_minus_alpha_bar_t * eps, eps

def p_losses(model, x_0, t, epoch_idx):
    """
    Compute the loss = MSE( noise_pred, noise ) for a random t.
    """
    x_t, noise = q_sample(x_0, t)
    noise_pred = model(x_t, t, epoch_idx)
    return nn.MSELoss()(noise_pred, noise)

# ----------------------------
# 5. Training
# ----------------------------
def train_diffusion(model, steps=TRAIN_STEPS):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    for step in range(steps):
        # sample real x_0 from dataset
        x_0, epoch_idx = sample_data(BATCH_SIZE)

        # pick a random diffusion time t in [0..DIFFUSION_STEPS-1]
        t = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), device=DEVICE)

        loss = p_losses(model, x_0, t, epoch_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:4d} / {steps}, loss={loss.item():.4f}")

    print("Training complete!")

# ----------------------------
# 6. Sampling (Reverse Diffusion)
# ----------------------------
@torch.no_grad()
def p_sample(model, x_t, t, epoch_idx):
    """
    One step of reverse diffusion:
    p(x_{t-1}|x_t) = N(mu, sigma^2 I),
    where mu = 1/sqrt(alpha_t) * ( x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t)* e_theta(x_t, t) )
    """
    betat = betas[t]
    alphat = alphas[t]
    alpha_bar_t = alpha_bars[t]

    sqrt_one_over_alpha_t = 1.0 / alphat.sqrt()
    sqrt_inv_alpha_bar_t = 1.0 / (alpha_bar_t.sqrt())
    # model predicts noise
    eps_theta = model(x_t, torch.tensor([t]*x_t.size(0), device=DEVICE), epoch_idx)

    coeff = (1 - alphat) / (1 - alpha_bar_t).sqrt()
    mean = sqrt_one_over_alpha_t * (x_t - coeff * eps_theta)

    if t > 0:
        z = torch.randn_like(x_t)
        sigma_t = betat.sqrt()
        sample = mean + sigma_t * z
    else:
        # at t=0, no noise
        sample = mean
    return sample

@torch.no_grad()
def p_sample_loop(model, epoch_idx, shape=(1, DIM_X)):
    """
    Start from pure noise x_T, and sample down to x_0.
    Returns x_0.
    """
    x_t = torch.randn(shape, device=DEVICE)
    for i in reversed(range(DIFFUSION_STEPS)):
        x_t = p_sample(model, x_t, i, epoch_idx)
    return x_t

# ----------------------------
# 7. Evaluation Utilities
# ----------------------------
def generate_true_samples(epoch, num_samples):
    """Generate samples from the true distribution for a given epoch."""
    if epoch < 10:
        mean = torch.tensor([-5.0, 5.0], device=DEVICE)
    else:
        mean = torch.tensor([5.0, 5.0], device=DEVICE)

    # Sample from Normal(mean, DATA_STD^2 * I)
    true_samples = mean + DATA_STD * torch.randn((num_samples, DIM_X), device=DEVICE)
    return true_samples.cpu().numpy()

def compute_distribution_metrics(generated_samples, true_samples):
    """Compute metrics to compare generated and true distributions."""
    gen_mean = np.mean(generated_samples, axis=0)
    true_mean = np.mean(true_samples, axis=0)

    gen_std = np.std(generated_samples, axis=0)
    true_std = np.std(true_samples, axis=0)

    mean_error = np.linalg.norm(gen_mean - true_mean)
    std_error = np.linalg.norm(gen_std - true_std)

    return {
        'generated_mean': gen_mean,
        'true_mean': true_mean,
        'mean_error': mean_error,
        'generated_std': gen_std,
        'true_std': true_std,
        'std_error': std_error
    }

# ----------------------------
# 8. Main Execution
# ----------------------------
def main():
    # Create model
    model = DiffusionModel(dim_x=DIM_X, embed_dim=EMBED_DIM).to(DEVICE)
    print(model)

    # Train the diffusion model
    train_diffusion(model, steps=TRAIN_STEPS)

    # Now let's sample from the model, conditioning on each epoch in [0..19],
    # and see if we recover the correct shift in means.
    model.eval()

    num_samples = 1000
    all_samples = []
    all_metrics = []

    # Generate samples for each epoch
    for e in range(EPOCHS_TOTAL):
        # We'll make a batch of epoch indices
        epoch_idx = torch.tensor([e]*num_samples, device=DEVICE)
        # sample from the model
        x_0_samples = p_sample_loop(model, epoch_idx, shape=(num_samples, DIM_X))
        # move to CPU for analysis
        x_0_samples_cpu = x_0_samples.cpu().numpy()
        all_samples.append(x_0_samples_cpu)

        # Generate true samples and compute metrics
        true_samples = generate_true_samples(e, num_samples)
        metrics = compute_distribution_metrics(x_0_samples_cpu, true_samples)
        all_metrics.append(metrics)

        print(f"Epoch {e:2d}: Generated mean = ({metrics['generated_mean'][0]:.2f}, {metrics['generated_mean'][1]:.2f}), "
              f"True mean = ({metrics['true_mean'][0]:.2f}, {metrics['true_mean'][1]:.2f}), "
              f"Mean error = {metrics['mean_error']:.4f}")

    # Visualization - compare epochs from first half and second half
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot epochs from first half (0-9) and second half (10-19)
    epochs_to_plot = [5, 15]  # From first half and second half

    for i, ep in enumerate(epochs_to_plot):
        # Generate samples
        samples_ep = all_samples[ep]
        true_samples = generate_true_samples(ep, num_samples)
        metrics = all_metrics[ep]

        # Plot in the first row
        ax = axes[0, i]
        ax.scatter(true_samples[:,0], true_samples[:,1], alpha=0.2, s=5, color='orange', label='True')
        ax.scatter(samples_ep[:,0], samples_ep[:,1], alpha=0.2, s=5, color='blue', label='Generated')

        # Add mean points
        ax.scatter([metrics['true_mean'][0]], [metrics['true_mean'][1]], color='red', s=100, marker='x', label='True Mean')
        ax.scatter([metrics['generated_mean'][0]], [metrics['generated_mean'][1]], color='green', s=100, marker='+', label='Generated Mean')

        ax.set_title(f"Epoch {ep} - Distribution Comparison")
        ax.set_xlim([-10, 10])
        ax.set_ylim([0, 10])
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left')

        # Add annotation showing mean values
        ax.annotate(f"True mean: ({metrics['true_mean'][0]:.2f}, {metrics['true_mean'][1]:.2f})\n"
                    f"Gen mean: ({metrics['generated_mean'][0]:.2f}, {metrics['generated_mean'][1]:.2f})\n"
                    f"Error: {metrics['mean_error']:.4f}",
                    xy=(0.05, 0.95), xycoords='axes fraction', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # Plot mean errors for all epochs in the second row
    mean_errors = [metrics['mean_error'] for metrics in all_metrics]
    ax = axes[1, 0]
    ax.bar(range(EPOCHS_TOTAL), mean_errors)
    ax.set_title("Mean Error by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Error")
    ax.grid(alpha=0.3)

    # Highlight first half vs second half
    ax.axvspan(0, 9.5, alpha=0.2, color='blue')
    ax.axvspan(9.5, 19.5, alpha=0.2, color='orange')
    ax.annotate("First Distribution\n(-5, 5)", xy=(4, max(mean_errors)*0.9), ha='center')
    ax.annotate("Second Distribution\n(5, 5)", xy=(14, max(mean_errors)*0.9), ha='center')

    # Plot std errors
    std_errors = [metrics['std_error'] for metrics in all_metrics]
    ax = axes[1, 1]
    ax.bar(range(EPOCHS_TOTAL), std_errors)
    ax.set_title("Standard Deviation Error by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Std Error")
    ax.grid(alpha=0.3)

    # Highlight first half vs second half
    ax.axvspan(0, 9.5, alpha=0.2, color='blue')
    ax.axvspan(9.5, 19.5, alpha=0.2, color='orange')

    plt.tight_layout()
    plt.savefig('diffusion_temporal_recall.png')
    plt.show()

if __name__ == "__main__":
    main()
