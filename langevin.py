import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from tqdm import tqdm


def gradient_log_gaussian(x: np.ndarray):
    """
    x of shape (N, features)
    """
    assert x.ndim == 2
    res = (-cov_inv @ (x - mu)[..., None]).squeeze(2)
    assert res.shape == x.shape
    return res


def plot_gaussian_ellipse(ax, mu: np.ndarray, cov: np.ndarray,
                          plot_eigenvectors: bool = True, color: str = 'orange'):
    theta = np.linspace(0, 2 * np.pi, 100)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    P = eigenvectors[:, ::-1]

    for l in range(1, 3):
        x = l * np.cos(theta) * np.sqrt(eigenvalues[-1])
        y = l * np.sin(theta) * np.sqrt(eigenvalues[-2])
        ellipse = P @ np.array([x, y]) + mu[:, None]
        ax.plot(ellipse[0], ellipse[1], color)

    if plot_eigenvectors:
        for idx in range(1, 3):
            ax.arrow(mu[0], mu[1],
                     dx=math.sqrt(eigenvalues[-idx]) * eigenvectors[0, -idx],
                     dy=math.sqrt(eigenvalues[-idx]) * eigenvectors[1, -idx],
                     head_width=0.1,
                     length_includes_head=True,
                     )


np.random.seed(0)
n_steps = 5000
# n_steps = 15
eps = 0.001
n_samples = 750

fig = plt.figure()
scatter, = plt.plot([], [], 'o', zorder=-1)
plt.gca().set_xlim([-5, 7])
plt.gca().set_ylim([-5, 7])
plt.gca().set_title('Langevin Sampling')
plt.gca().set_xlabel('x1')
plt.gca().set_ylabel('x2')
dpi = 100
metadata = dict(title='Langevin Sampling')
writer = PillowWriter(fps=15, metadata=metadata)

cov = np.array(
    [
        [1.0, -0.5],
        [-0.5, 1.0],
    ])
mu = np.array([4, 3])

cov_inv = np.linalg.inv(cov)

brownian_noise = math.sqrt(2 * eps) * np.random.randn(n_steps, n_samples, len(mu))
x = np.random.randn(n_samples, 2)

plot_gaussian_ellipse(plt.gca(), np.zeros(2), np.eye(2))
plot_gaussian_ellipse(plt.gca(), mu, cov)

with writer.saving(fig, 'langevin.gif', dpi):
    for i in tqdm(range(n_steps)):
        x += eps * gradient_log_gaussian(x) + brownian_noise[i]
        scatter.set_data(x[:, 0], x[:, 1])
        if i % 7 == 0:
            writer.grab_frame()

