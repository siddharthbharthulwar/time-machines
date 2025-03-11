'''
Demonstration of a 2D diffusion model that uses class conditioning.
We treat each Gaussian distribution (points1 & points2) as a separate class (0 or 1).
'''

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

# ------------------------ Conditioned MLP ------------------------ #
class SimplePointMLPCond(nn.Module):
    """
    A simple MLP for predicting noise in 2D points,
    conditioned on a time step `t` and a class label `c`.
    """
    def __init__(
        self,
        point_dim: int = 2,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 16,
        hidden_dim: int = 256,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Embedding for the time step
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Embedding for the label (class)
        self.label_embed = nn.Embedding(num_classes, cond_embed_dim)
        
        # Main MLP takes: [point_dim + time_embed_dim + cond_embed_dim]
        self.net = nn.Sequential(
            nn.Linear(point_dim + time_embed_dim + cond_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, point_dim),
        )

    def forward(self, x, t, c):
        """
        Args:
            x: shape (batch_size, point_dim)
            t: shape (batch_size,) for time steps
            c: shape (batch_size,) for class labels
        
        Returns:
            noise_pred: shape (batch_size, point_dim)
        """
        # Create time embeddings
        t_emb = self.time_embed(t.unsqueeze(-1))  # (batch_size, time_embed_dim)
        
        # Create label embeddings
        c_emb = self.label_embed(c)               # (batch_size, cond_embed_dim)
        
        # Concatenate [x, t_emb, c_emb]
        x_t = torch.cat([x, t_emb, c_emb], dim=-1)
        
        # Predict noise
        noise_pred = self.net(x_t)
        return noise_pred

# ------------------------ Diffusion Class (with conditioning) ------------------------ #
class PointDiffusion:
    def __init__(
        self,
        point_dim: int = 2,
        n_steps: int = 1000,
        device: str = "cuda",
        num_classes: int = 2
    ):
        self.point_dim = point_dim
        self.n_steps = n_steps
        self.device = device
        
        # Define noise schedule (linear beta schedule)
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Create noise prediction network (conditioned)
        self.model = SimplePointMLPCond(
            point_dim=point_dim,
            time_embed_dim=64,
            cond_embed_dim=16,
            hidden_dim=256,
            num_classes=num_classes
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


    def diffuse_points(self, x_0, t):
        """
        Add noise to points according to diffusion schedule at time t.
        
        Args:
            x_0: shape (batch_size, point_dim)
            t: shape (batch_size,) with integer values in [0, n_steps)
        
        Returns:
            x_t: The noisy points at time t
            eps: The noise that was added
        """
        x_0 = x_0.to(self.device)
        
        a_bar = self.alpha_bar[t]  # (batch_size,)
        a_bar = a_bar.view(-1, 1)  # reshape for broadcasting
        
        # Sample noise
        eps = torch.randn_like(x_0, device=self.device)
        
        # Diffuse
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eps
        return x_t, eps
    
    def train_step(self, x_0, c):
        """
        Single training step with diffusion denoising.
        
        Args:
            x_0: shape (batch_size, point_dim) real points
            c:   shape (batch_size,) class labels for conditioning
        
        Returns:
            denoising_loss.item()
        """
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x_0.shape[0],), device=self.device)
        
        # Add noise
        x_t, noise = self.diffuse_points(x_0, t)
        
        # Predict noise, passing in the time (normalized) and the label
        noise_pred = self.model(x_t, t.float() / self.n_steps, c)
        
        # Standard denoising loss
        denoising_loss = nn.MSELoss()(noise_pred, noise)
        
        # Optimize
        self.optimizer.zero_grad()
        denoising_loss.backward()
        self.optimizer.step()
        
        return denoising_loss.item()

    @torch.no_grad()
    def sample(self, n_points: int, shape: int, c: int, t: int = None):
        """
        Sample new points from the diffusion model, conditioned on class c.
        
        Args:
            n_points: how many points to sample
            shape: dimensionality of each point (2D -> shape=2)
            c: the integer class label
            t: (optional) starting diffusion time index (if None, defaults to n_steps-1)
        
        Returns:
            x: shape (n_points, shape) sampled points
        """
        if t is None:
            t = self.n_steps - 1
        
        # Start from random noise
        x = torch.randn(n_points, shape, device=self.device)
        
        # Create a label vector of size n_points
        c_tensor = torch.full((n_points,), fill_value=c, device=self.device, dtype=torch.long)
        
        # Gradually denoise from t down to 0
        for step_t in range(t, -1, -1):
            t_tensor = torch.ones(n_points, device=self.device) * step_t
            
            # Predict noise
            eps_theta = self.model(x, t_tensor.float() / self.n_steps, c_tensor)
            
            alpha = self.alpha[step_t]
            alpha_bar = self.alpha_bar[step_t]
            
            if step_t > 0:
                z = torch.randn_like(x)
            else:
                z = 0
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
            ) + torch.sqrt(self.beta[step_t]) * z
            
        return x

# ------------------------ Datasets ------------------------ #
def gaussian_dataset(n=8000, mean=(0.0, 0.0), variance=1.0, device="cpu"):
    """
    Creates a dataset of points sampled from a 2D Gaussian distribution
    """
    mean_tensor = torch.tensor(mean, device=device)
    std_dev = math.sqrt(variance)
    
    points = torch.randn(n, 2, device=device)
    points = points * std_dev + mean_tensor
    return points

class GradualShiftDataset(Dataset):
    """
    Dataset that dynamically samples from two point distributions
    based on a configurable mix ratio.
    We also assign labels: 0 for points1, 1 for points2.
    """
    def __init__(self, points1, points2, label1=0, label2=1):
        """
        Args:
            points1 (torch.Tensor): shape (N,2), distribution #1
            points2 (torch.Tensor): shape (N,2), distribution #2
            label1, label2: integer labels for conditioning
        """
        self.points1 = points1
        self.points2 = points2
        self.label1 = label1
        self.label2 = label2
        
        self.mix_ratio = 0.0  # Probability of sampling from points2
        self.num_points = len(points1)
        
    def __len__(self):
        return self.num_points
    
    def __getitem__(self, idx):
        """
        Sample a point based on current mix ratio, plus its label.
        """
        if torch.rand(1).item() < self.mix_ratio:
            # from distribution #2
            random_idx = torch.randint(0, len(self.points2), (1,)).item()
            return self.points2[random_idx], torch.tensor(self.label2, dtype=torch.long)
        else:
            # from distribution #1
            random_idx = torch.randint(0, len(self.points1), (1,)).item()
            return self.points1[random_idx], torch.tensor(self.label1, dtype=torch.long)
    
    def set_mix_ratio(self, mix_ratio):
        self.mix_ratio = max(0.0, min(1.0, mix_ratio))

# ------------------------ Main ------------------------ #
if __name__ == "__main__":
    # Detect how many GPUs are available
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device = "cuda:1"
    elif num_gpus == 1:
        device = "cuda:0"
    else:
        device = "cpu"

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create two Gaussian clusters (two classes)
    points1 = gaussian_dataset(n=8000, mean=(-1.0, -1.0), variance=0.1, device=device)
    points2 = gaussian_dataset(n=8000, mean=(1.0, 1.0), variance=0.1, device=device)

    dataset = GradualShiftDataset(points1, points2, label1=0, label2=1)
    n_epochs = 100

    # Create conditional diffusion model (2 classes)
    diffusion_model = PointDiffusion(
        n_steps=50,
        point_dim=2,
        device=device,
        num_classes=2
    )

    pbar = tqdm(range(n_epochs), desc="Training (Conditional) Diffusion Model")

    for epoch in pbar:
        # Gradually shift how often we sample from distribution #2
        dataset.set_mix_ratio(epoch / n_epochs)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for (batch_points, batch_labels) in dataloader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)
            
            loss = diffusion_model.train_step(batch_points, batch_labels)
            pbar.set_description(
                f"Epoch {epoch} | Loss: {loss:.4f} | Mix ratio: {dataset.mix_ratio:.2f}"
            )

        # Generate and save samples every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            # We'll sample from BOTH classes
            with torch.no_grad():
                samples_class0 = diffusion_model.sample(1000, 2, c=0, t=49)
                samples_class1 = diffusion_model.sample(1000, 2, c=1, t=49)

            # Plot real data vs generated
            plt.figure(figsize=(10, 5))
            
            # Subplot for class 0
            plt.subplot(1, 2, 1)
            plt.title(f"Epoch {epoch} - Class 0")
            # We'll collect real data from class 0 only
            real_c0 = [dataset[i][0].cpu().numpy() for i in range(1000) if dataset[i][1].item() == 0]
            real_c0 = np.array(real_c0)
            if len(real_c0) > 0:
                plt.scatter(real_c0[:,0], real_c0[:,1], alpha=0.5, label="Real (class 0)")
            plt.scatter(samples_class0[:, 0].cpu(), samples_class0[:, 1].cpu(),
                        alpha=0.5, label="Generated (class 0)")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.legend()

            # Subplot for class 1
            plt.subplot(1, 2, 2)
            plt.title(f"Epoch {epoch} - Class 1")
            # We'll collect real data from class 1 only
            real_c1 = [dataset[i][0].cpu().numpy() for i in range(1000) if dataset[i][1].item() == 1]
            real_c1 = np.array(real_c1)
            if len(real_c1) > 0:
                plt.scatter(real_c1[:,0], real_c1[:,1], alpha=0.5, label="Real (class 1)")
            plt.scatter(samples_class1[:, 0].cpu(), samples_class1[:, 1].cpu(),
                        alpha=0.5, label="Generated (class 1)")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"samples_epoch_{epoch}.png")
            plt.close()
