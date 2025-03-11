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


class SimplePointMLP(nn.Module):
    def __init__(self, point_dim, time_embed_dim=64, hidden_dim=256):  # increased dimensions
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(point_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),  # added another layer
            nn.SiLU(),
            nn.Linear(hidden_dim, point_dim)
        )

    def forward(self, x, t):
        # Create time embeddings
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # Concatenate point data with time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # Predict noise
        return self.net(x_t)
    
class PointDiffusion:
    def __init__(self, point_dim, n_steps=1000, device="cuda"):
        self.point_dim = point_dim
        self.n_steps = n_steps
        self.device = device
        
        # Define noise schedule (linear beta schedule)
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Create noise prediction network
        self.model = SimplePointMLP(point_dim=point_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)



    def diffuse_points(self, x_0, t):
        """Add noise to points according to diffusion schedule"""
        # Ensure x_0 is on the correct device
        x_0 = x_0.to(self.device)
        
        a_bar = self.alpha_bar[t]
        
        # Reshape for broadcasting
        a_bar = a_bar.view(-1, 1)
        
        # Sample noise on the same device as x_0
        eps = torch.randn_like(x_0, device=self.device)
        
        # Create noisy points
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eps
        
        return x_t, eps
    
    def train_step(self, x_0):
        """
        Single training step with diffusion denoising.
        
        Args:
            x_0: Input data points
            k_steps: Number of denoising steps (not used in this implementation)
            lambda_equiv: Weight for equivariance loss (not used in this implementation)
            group_action_fn: Function that applies the group action (not used in this implementation)
            group_action_params: Parameters for the group action (not used in this implementation)
            timestep_mask: Specifies which timesteps to apply equivariance loss to (not used in this implementation)
                           
        Note on timesteps:
        - In diffusion models, t=0 is the END of the denoising process (clean data)
        - t=n_steps-1 is the BEGINNING of the denoising process (pure noise)
        - The forward process adds noise from t=0 to t=n_steps-1
        - The reverse process (denoising) goes from t=n_steps-1 down to t=0
        
        Returns:
            float: The denoising loss value
        """
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x_0.shape[0],)).to(self.device)
        
        # Add noise to points
        x_t, noise = self.diffuse_points(x_0, t)

        # Calculate standard denoising loss (using single step prediction)
        noise_pred = self.model(x_t, t.float() / self.n_steps)
        denoising_loss = nn.MSELoss()(noise_pred, noise)

        # Optimize
        self.optimizer.zero_grad()
        denoising_loss.backward()
        self.optimizer.step()

        return denoising_loss.item()

    @torch.no_grad()
    def sample(self, n_points, shape, t=0):
        """Sample new points from the diffusion model"""
        # Start from random noise
        x = torch.randn(n_points, shape).to(self.device)
        
        # Gradually denoise
        for t in range(t, -1, -1):
            t_tensor = torch.ones(n_points).to(self.device) * t
            
            # Predict noise
            eps_theta = self.model(x, t_tensor.float() / self.n_steps)
            
            # Get alpha values for current timestep
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            
            # Sample noise for stochastic sampling (except at t=0)
            z = torch.randn_like(x) if t > 0 else 0
            
            # Update point estimates
            x = 1 / torch.sqrt(alpha) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
            ) + torch.sqrt(self.beta[t]) * z
            
        return x
    
@torch.no_grad()
def compute_vector_field_error(field1, field2):
    """
    Compute average error between two vector fields using component-wise difference.
    
    Args:
        field1, field2: Arrays of shape (grid_size, grid_size, 2) containing vector components
    
    Returns:
        Scalar value representing average error across all grid points
    """
    # Compute component-wise differences
    diff_field = field1 - field2
    
    # Compute magnitude of difference vectors
    error_map = np.sqrt(diff_field[..., 0]**2 + diff_field[..., 1]**2)
    
    # Average across all points in the grid
    mean_error = np.mean(error_map)
    
    return mean_error
    
@torch.no_grad()
def compute_score_field(diffusion_model: PointDiffusion, 
                       x_range: Tuple[float, float]=(-1.5, 1.5), 
                       y_range: Tuple[float, float]=(-1.5, 1.5),
                       grid_size: int=50,
                       timestep: int=500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute score vectors over a grid of points.
    """
    # Create grid points
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx, yy], axis=-1)
    
    # Convert to tensor
    grid_tensor = torch.from_numpy(grid_points.reshape(-1, 2)).float().to(diffusion_model.device)

    # Ensure timestep is within bounds
    t = torch.ones(grid_tensor.shape[0], device=diffusion_model.device) * (timestep % diffusion_model.n_steps)
    
    # Compute scores using noise prediction
    with torch.no_grad():
        eps_theta = diffusion_model.model(grid_tensor, t.float() / diffusion_model.n_steps)
        alpha_bar_t = diffusion_model.alpha_bar[t.long()]
        scores = -eps_theta / torch.sqrt(1 - alpha_bar_t.view(-1, 1))
    
    # Reshape scores back to grid
    score_x = scores[:, 0].cpu().numpy().reshape(grid_size, grid_size)
    score_y = scores[:, 1].cpu().numpy().reshape(grid_size, grid_size)
    
    return xx, yy, score_x, score_y

def gaussian_dataset(n=8000, mean=(0.0, 0.0), variance=1.0, device="cpu"):
    """Creates a dataset of points sampled from a 2D Gaussian distribution
    
    Args:
        n (int): Number of points to generate
        mean (tuple): Mean of the Gaussian distribution (x, y)
        variance (float): Variance of the Gaussian distribution
        device (str): Device to create tensors on
    
    Returns:
        torch.Tensor: Tensor of shape (n, 2) containing 2D points
    """
    # Convert mean to tensor
    mean_tensor = torch.tensor(mean, device=device)
    
    # Create covariance matrix (assuming isotropic Gaussian)
    std_dev = math.sqrt(variance)
    
    # Sample from standard normal distribution
    points = torch.randn(n, 2, device=device)
    
    # Scale by standard deviation and shift by mean
    points = points * std_dev + mean_tensor
    
    return points

class GradualShiftDataset(torch.utils.data.Dataset):
    """
    Dataset that dynamically samples from two point distributions
    based on a configurable mix ratio.
    """
    def __init__(self, points1, points2):
        """
        Args:
            points1 (torch.Tensor): First distribution of points
            points2 (torch.Tensor): Second distribution of points
        """
        self.points1 = points1
        self.points2 = points2
        self.mix_ratio = 0.0  # Start with all points from points1
        self.num_points = len(points1)  # Assuming both have same length
        
    def __len__(self):
        return self.num_points
    
    def __getitem__(self, idx):
        """Sample a point based on current mix ratio"""
        # Randomly determine whether to sample from points1 or points2
        if torch.rand(1).item() < self.mix_ratio:
            # Sample from points2
            random_idx = torch.randint(0, len(self.points2), (1,)).item()
            return self.points2[random_idx]
        else:
            # Sample from points1
            random_idx = torch.randint(0, len(self.points1), (1,)).item()
            return self.points1[random_idx]
    
    def set_mix_ratio(self, mix_ratio):
        """Update the mix ratio between the two distributions"""
        self.mix_ratio = max(0.0, min(1.0, mix_ratio))  # Ensure it's between 0 and 1


if __name__ == "__main__":

    # Set fixed seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    points1 = gaussian_dataset(n=8000, mean=(-1.0, -1.0), variance=0.1, device=device)
    points2 = gaussian_dataset(n=8000, mean=(1.0, 1.0), variance=0.1, device=device)

    # Create dataset
    dataset = GradualShiftDataset(points1, points2)
    
    n_epochs = 100

    pbar = tqdm(range(n_epochs), desc="Training gradual shift model")

    # initialize diffusion model
    diffusion_model = PointDiffusion(
        n_steps=50,
        point_dim=2,  # 2D points
    )
    for epoch in pbar:
        dataset.set_mix_ratio(epoch / n_epochs)

        # Create dataloader for the current epoch
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for batch in dataloader:
            loss = diffusion_model.train_step(batch)
            pbar.set_description(f"Training gradual shift model | Loss: {loss:.4f} | Mix ratio: {dataset.mix_ratio:.2f}")
        # Generate and save samples every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                # Generate samples
                samples = diffusion_model.sample(1000, 2, t=49)
            # Plot samples along with the dataset points
            plt.figure(figsize=(10, 10))
            
            # Plot dataset points
            current_points = torch.stack([dataset[i] for i in range(1000)])
            plt.scatter(current_points[:, 0].cpu(), current_points[:, 1].cpu(),
                        alpha=0.5, label=f"Data (mix={dataset.mix_ratio:.2f})")
            
            # Plot generated samples
            plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(),
                        alpha=0.5, label="Generated")
            
            plt.legend()
            plt.title(f"Epoch {epoch}: Generated vs Real Points")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.savefig(f"samples_epoch_{epoch}.png")
            plt.close()
            
            

    
        
        




