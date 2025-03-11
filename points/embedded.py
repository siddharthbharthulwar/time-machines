''' 
This script does unconditional image generation on MNIST using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239
'''

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import gc
import math
import imageio
from mpl_toolkits.mplot3d import Axes3D


class FiLMBlock(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    Given a conditioning vector (e.g. domain/time embedding),
    we produce gamma, beta to scale & shift the input features.
    """
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        """
        x:    (batch, hidden_dim)
        cond: (batch, cond_dim) -- combined domain/time embedding
        """
        # Project cond -> gamma, beta
        gamma_ = self.gamma(cond)  # (batch, hidden_dim)
        beta_  = self.beta(cond)   # (batch, hidden_dim)
        # FiLM operation
        return x * (1 + gamma_) + beta_
    
class FiLMPointMLP(nn.Module):
    def __init__(
        self,
        point_dim: int,
        diffusion_time_embed_dim: int = 64,
        domain_embed_dim: int = 32,
        hidden_dim: int = 256
    ):
        """
        point_dim: size of your data (e.g., 2D points)
        diffusion_time_embed_dim: dimension of the embedding for diffusion t
        domain_embed_dim: dimension of the embedding for domain/epoch
        hidden_dim: base hidden dimension
        """
        super().__init__()

        # 1) Embedding layers for time and domain
        self.time_embed = nn.Sequential(
            nn.Linear(1, diffusion_time_embed_dim),
            nn.SiLU(),
            nn.Linear(diffusion_time_embed_dim, diffusion_time_embed_dim),
        )
        self.domain_embed = nn.Sequential(
            nn.Linear(1, domain_embed_dim),
            nn.SiLU(),
            nn.Linear(domain_embed_dim, domain_embed_dim),
        )

        # We'll combine them into a single "conditioning" vector
        self.cond_dim = diffusion_time_embed_dim + domain_embed_dim

        # 2) Define the feed‐forward layers
        #    with FiLM at each layer
        self.fc1 = nn.Linear(point_dim, hidden_dim)
        self.film1 = FiLMBlock(cond_dim=self.cond_dim, hidden_dim=hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMBlock(cond_dim=self.cond_dim, hidden_dim=hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.film3 = FiLMBlock(cond_dim=self.cond_dim, hidden_dim=hidden_dim)

        self.out = nn.Linear(hidden_dim, point_dim)

    def forward(self, x, t_diffusion, t_domain):
        """
        x:          (batch, point_dim) 
        t_diffusion (batch,) -> in [0..1]
        t_domain    (batch,) -> in [0..1]
        """
        # Expand dimension so linear layers see shape (B,1)
        t_diffusion = t_diffusion.unsqueeze(-1)
        t_domain    = t_domain.unsqueeze(-1)

        # Create embeddings
        t_emb = self.time_embed(t_diffusion)    # (batch, time_embed_dim)
        d_emb = self.domain_embed(t_domain)     # (batch, domain_embed_dim)
        
        # Combine them for FiLM
        cond = torch.cat([t_emb, d_emb], dim=-1)  # (batch, cond_dim)

        # Pass x through layers with FiLM
        x = self.fc1(x)
        x = self.film1(x, cond)
        x = F.silu(x)

        x = self.fc2(x)
        x = self.film2(x, cond)
        x = F.silu(x)

        x = self.fc3(x)
        x = self.film3(x, cond)
        x = F.silu(x)

        # Final output
        x = self.out(x)
        return x




# -----------------------------------------------------------------------------------
# 1. Simple MLP with both diffusion-time embedding + domain-time (or "epoch") embedding
# -----------------------------------------------------------------------------------
class SimplePointMLP(nn.Module):
    def __init__(self, 
                 point_dim: int, 
                 diffusion_time_embed_dim: int = 64, 
                 domain_embed_dim: int = 32, 
                 hidden_dim: int = 256):
        """
        point_dim: dimension of your data (e.g. 2D points)
        diffusion_time_embed_dim: dimension of the embedding for the diffusion timestep
        domain_embed_dim: dimension of the embedding for the 'domain timestamp' (epoch, etc.)
        hidden_dim: hidden dimension for MLP
        """
        super().__init__()
        
        # Embedding for diffusion time (t ranges from 0..1 in your code)
        self.time_embed = nn.Sequential(
            nn.Linear(1, diffusion_time_embed_dim),
            nn.SiLU(),
            nn.Linear(diffusion_time_embed_dim, diffusion_time_embed_dim),
        )
        
        # Embedding for domain timestamp (e.g., epoch, distribution shift ID, etc.)
        self.domain_embed = nn.Sequential(
            nn.Linear(1, domain_embed_dim),
            nn.SiLU(),
            nn.Linear(domain_embed_dim, domain_embed_dim),
        )
        
        # Combine point_dim + diffusion time embedding + domain embedding -> hidden layers
        self.net = nn.Sequential(
            nn.Linear(point_dim + diffusion_time_embed_dim + domain_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, point_dim)
        )

    def forward(self, x, t_diffusion, t_domain):
        """
        x:           (batch, point_dim) 
        t_diffusion: (batch,) float from 0..1 for diffusion step
        t_domain:    (batch,) float from 0..1 for 'domain timestamp' or epoch
        """
        # Expand dimension so linear layers see shape (B, 1)
        t_diffusion = t_diffusion.unsqueeze(-1)
        t_domain    = t_domain.unsqueeze(-1)
        
        # Create embeddings
        t_emb      = self.time_embed(t_diffusion)
        domain_emb = self.domain_embed(t_domain)

        # Concatenate x, diffusion time embedding, domain embedding
        x_t = torch.cat([x, t_emb, domain_emb], dim=-1)
        
        # Predict noise
        return self.net(x_t)

# -----------------------------------------------------------------------------------
# 2. Diffusion class that uses the above MLP and expects a domain-time input
# -----------------------------------------------------------------------------------
class PointDiffusion:
    def __init__(self, point_dim, n_steps=1000, device="cuda"):
        self.point_dim = point_dim
        self.n_steps = n_steps
        self.device = device
        
        # Define noise schedule (linear beta schedule)
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Create noise prediction network: note that we also have domain_embed_dim
        self.model = FiLMPointMLP(
            point_dim=point_dim,
            diffusion_time_embed_dim=64,
            domain_embed_dim=64,    # you can tune this
            hidden_dim=256
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def diffuse_points(self, x_0, t):
        """
        Forward diffusion: x_0 -> x_t
        t is a tensor of shape (batch,) with integer timesteps in [0, n_steps-1]
        """
        x_0 = x_0.to(self.device)        
        a_bar = self.alpha_bar[t].view(-1, 1)  # shape (batch,1)
        
        eps = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eps
        return x_t, eps

    def train_step(self, x_0, domain_t):
        """
        Single training step for diffusion denoising with domain-time conditioning.
        
        Args:
            x_0:       (batch_size, point_dim)
            domain_t:  (batch_size,) domain timestamps (e.g. normalized epoch or mix ratio)
        """
        # Sample random diffusion timesteps for each item in batch
        t = torch.randint(0, self.n_steps, (x_0.shape[0],)).to(self.device)
        
        # Diffuse points
        x_t, noise = self.diffuse_points(x_0, t)

        # Predict noise. 
        # - normalize t by n_steps to get a 0..1 float
        # - domain_t is already in [0..1], or however you define it
        noise_pred = self.model(x_t, t.float() / self.n_steps, domain_t)

        denoising_loss = F.mse_loss(noise_pred, noise)

        # Optimize
        self.optimizer.zero_grad()
        denoising_loss.backward()
        self.optimizer.step()

        return denoising_loss.item()

    @torch.no_grad()
    def sample(self, n_points, shape, domain_t_val=0.0):
        """
        Sample new points from the diffusion model, 
        optionally specifying a domain timestamp value (float) for ALL samples.
        """
        # Start from random noise
        x = torch.randn(n_points, shape).to(self.device)
        
        # We also need a domain timestamp for each sample
        domain_t = torch.ones(n_points, device=self.device) * domain_t_val
        
        for t in range(self.n_steps-1, -1, -1):
            t_tensor = torch.ones(n_points, device=self.device) * t
            
            # Predict noise
            eps_theta = self.model(x, t_tensor.float()/self.n_steps, domain_t)
            
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
            ) + torch.sqrt(self.beta[t]) * z
        return x
    

# -----------------------------------------------------------------------------------
# 3. Example dataset that transitions from one Gaussian to another
# -----------------------------------------------------------------------------------
def gaussian_dataset(n=8000, mean=(0.0, 0.0), variance=1.0, device="cpu"):
    mean_tensor = torch.tensor(mean, device=device)
    std_dev = math.sqrt(variance)
    points = torch.randn(n, 2, device=device) * std_dev + mean_tensor
    return points

class GradualShiftDataset(torch.utils.data.Dataset):
    """
    Dataset that dynamically samples from two point distributions
    based on a configurable mix ratio from [0..1].
    """
    def __init__(self, points1, points2):
        super().__init__()
        self.points1 = points1
        self.points2 = points2
        self.mix_ratio = 0.0  
        self.num_points = len(points1)  # assume both have same length
    
    def __len__(self):
        return self.num_points
    
    def __getitem__(self, idx):
        if torch.rand(1).item() < self.mix_ratio:
            # Sample from points2
            random_idx = torch.randint(0, len(self.points2), (1,)).item()
            return self.points2[random_idx]
        else:
            # Sample from points1
            random_idx = torch.randint(0, len(self.points1), (1,)).item()
            return self.points1[random_idx]
    
    def set_mix_ratio(self, mix_ratio):
        self.mix_ratio = max(0.0, min(1.0, mix_ratio))

# -----------------------------------------------------------------------------------
# 4. Main training loop: show how to feed domain-time (e.g., 'epoch ratio') 
#    to the diffusion model
# -----------------------------------------------------------------------------------
if __name__ == "__main__":

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = "./results/points/embedded"
    os.makedirs(save_dir, exist_ok=True)

    # Create two Gaussian distributions
    points1 = gaussian_dataset(n=8000, mean=(-5.0, -5.0), variance=0.1, device=device)
    points2 = gaussian_dataset(n=8000, mean=(5.0, 5.0), variance=0.1, device=device)

    # Create dataset
    dataset = GradualShiftDataset(points1, points2)
    
    n_epochs = 100
    # Initialize diffusion model with domain-time embedding
    diffusion_model = PointDiffusion(
        n_steps=50,   # fewer steps for quick demonstration
        point_dim=2,  # 2D data
        device=device
    )

    pbar = tqdm(range(n_epochs), desc="Training with domain-time embedding")
    
    # Track losses for plotting
    epoch_losses = []

    for epoch in pbar:
        # Example schedule for mix_ratio:
        mix_ratio = 0.0
        
        # Update the dataset's distribution mixing
        dataset.set_mix_ratio(mix_ratio)

        # Create a dataloader
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        # We'll define a "domain_time" for each sample as simply the current mix_ratio
        # but you could also do domain_time = epoch / n_epochs, or more advanced scheduling
        domain_t_value = 0

        # Track losses for this epoch
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x_0 = batch.to(device)
            
            # We must pass domain_t as shape (batch_size,)
            domain_t = torch.ones(x_0.size(0), device=device) * domain_t_value

            loss = diffusion_model.train_step(x_0, domain_t)
            epoch_loss += loss
            num_batches += 1
            
            pbar.set_description(
                f"Epoch {epoch} | Loss: {loss:.4f} | Mix ratio: {mix_ratio:.2f}"
            )
        
        # Store average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        # Generate and save samples every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            # We'll sample at the *same domain time* used for training in this epoch.
            # (You can also do multiple domain times and compare.)
            domain_t_for_sampling = 0

            with torch.no_grad():
                samples = diffusion_model.sample(n_points=1000, 
                                                 shape=2, 
                                                 domain_t_val=domain_t_for_sampling)
            
            # For illustration, let's also sample some actual dataset points
            # to compare real vs. generated distribution
            real_samples = torch.stack([dataset[i] for i in range(1000)])
            
            plt.figure(figsize=(6, 6))
            plt.scatter(
                real_samples[:, 0].cpu(),
                real_samples[:, 1].cpu(),
                alpha=0.4, label=f"Real data (mix={mix_ratio:.2f})"
            )
            plt.scatter(
                samples[:, 0].cpu(),
                samples[:, 1].cpu(),
                alpha=0.4, label="Generated"
            )
            plt.title(f"Epoch {epoch}: domain_t={domain_t_for_sampling:.2f}")
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"samples_epoch_{epoch}.png"))
            plt.close()
    
    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_epochs), epoch_losses, marker='o', markersize=3, linestyle='-')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()
    
    print(f"Training complete. Results saved to {save_dir}")

    # Create a GIF showing how the generated distribution changes with domain_t
    print("Creating GIF of generated distributions across domain times...")
    
    # Define the range of domain times to sample
    domain_t_values = np.linspace(0, 1, 20)  # 20 frames from domain_t=0 to domain_t=1
    
    # Create a directory for the frames if it doesn't exist
    frames_dir = os.path.join(save_dir, "domain_t_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate frames for each domain_t value
    frame_paths = []
    
    for i, domain_t in enumerate(domain_t_values):
        with torch.no_grad():
            samples = diffusion_model.sample(n_points=1000, 
                                             shape=2, 
                                             domain_t_val=domain_t)
        
        # Get real samples for comparison
        real_samples = torch.stack([dataset[i] for i in range(1000)])
        
        # Create the plot
        plt.figure(figsize=(6, 6))
        plt.scatter(
            real_samples[:, 0].cpu(),
            real_samples[:, 1].cpu(),
            alpha=0.4, label=f"Real data (mix={mix_ratio:.2f})"
        )
        plt.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            alpha=0.4, label="Generated"
        )
        plt.title(f"Generated distribution at domain_t={domain_t:.2f}")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        frame_paths.append(frame_path)
        
        print(f"Generated frame for domain_t={domain_t:.2f}")
    
    # Create the GIF from the frames
    print("Creating GIF from frames...")
    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    gif_path = os.path.join(save_dir, "domain_t_evolution.gif")
    imageio.mimsave(gif_path, frames, duration=300)  # 300ms per frame
    
    print(f"GIF saved to {gif_path}")