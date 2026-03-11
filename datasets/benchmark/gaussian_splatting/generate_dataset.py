#!/usr/bin/env python3
"""Generate 3D Gaussian Splatting benchmark dataset.

Forward model (3D Gaussian Splatting -- multi-view rendering):
    A 3D scene is represented as a collection of K anisotropic 3D Gaussians,
    each parameterised by (mu_k, Sigma_k, alpha_k, c_k):
        mu_k    -- 3D centre position
        Sigma_k -- 3x3 covariance matrix (orientation + scale)
        alpha_k -- opacity in [0, 1]
        c_k     -- RGB colour

    For a given camera pose P = (R, t, K):
        1. Project 3D means to 2D:  mu_2d = K @ (R @ mu_k + t)
        2. Project covariance:      Sigma_2d = J @ R @ Sigma_k @ R^T @ J^T
           where J is the Jacobian of the perspective projection
        3. Evaluate 2D Gaussian splat: G_2d(x; mu_2d, Sigma_2d)
        4. Alpha-composite front-to-back:
           C(x) = sum_k( T_k * alpha_k * G_2d_k(x) * c_k )
           where T_k = prod_{j<k}(1 - alpha_j * G_2d_j(x))

    Measurement = sparse multi-view observations with noise:
        y_v = C_v(x) + noise   for v in {view_1, ..., view_V}

Mismatch parameters (ThetaSpace):
    camera_pose_error    : rotation + translation perturbation on camera extrinsics
    depth_uncertainty    : error in depth ordering / z-buffer accuracy
    opacity_noise        : noise on per-Gaussian opacity values
    view_sparsity        : fraction of views withheld (more sparse = harder)

Ground truth phantoms (256x256):
    Simple 3D scenes: sphere arrangements, geometric primitives, overlapping
    ellipsoids rendered as 2D projections with volumetric alpha compositing.

Baseline:
    Direct alpha-blended rendering from noisy Gaussian parameters. ~22-28 dB.

Tiers:
    public : 12 samples (seeds from 0)
    dev    : 20 samples (seeds from 10000)
    hidden : 20 samples (seeds from 20000)

Usage:
    cd datasets/benchmark/gaussian_splatting
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# =============================================================================
# 3D Gaussian Splatting Camera Model
# =============================================================================

# Default camera intrinsics (pinhole camera, focal length in pixels)
FOCAL_LENGTH = 300.0        # focal length in pixels
CAMERA_DIST = 4.0           # camera distance from scene centre
N_VIEWS_DEFAULT = 8         # number of views for multi-view rendering
SCENE_RADIUS = 1.5          # bounding sphere radius of 3D scenes

# Mismatch spec ranges per tier
SPEC = {
    "public": {
        "camera_pose_error":  {"min": 0.005, "max": 0.02,  "unit": "radians+fraction"},
        "depth_uncertainty":  {"min": 0.005, "max": 0.02,  "unit": "relative"},
        "opacity_noise":      {"min": 0.01,  "max": 0.05,  "unit": "sigma"},
        "view_sparsity":      {"min": 0.0,   "max": 0.25,  "unit": "fraction_dropped"},
    },
    "dev": {
        "camera_pose_error":  {"min": 0.01,  "max": 0.05,  "unit": "radians+fraction"},
        "depth_uncertainty":  {"min": 0.01,  "max": 0.05,  "unit": "relative"},
        "opacity_noise":      {"min": 0.02,  "max": 0.08,  "unit": "sigma"},
        "view_sparsity":      {"min": 0.1,   "max": 0.4,   "unit": "fraction_dropped"},
    },
    "hidden": {
        "camera_pose_error":  {"min": 0.02,  "max": 0.10,  "unit": "radians+fraction"},
        "depth_uncertainty":  {"min": 0.02,  "max": 0.10,  "unit": "relative"},
        "opacity_noise":      {"min": 0.05,  "max": 0.15,  "unit": "sigma"},
        "view_sparsity":      {"min": 0.2,   "max": 0.5,   "unit": "fraction_dropped"},
    },
}


# =============================================================================
# Camera Utilities
# =============================================================================

def make_intrinsic(f: float, cx: float, cy: float) -> np.ndarray:
    """3x3 pinhole intrinsic matrix."""
    K = np.array([
        [f,  0, cx],
        [0,  f, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    return K


def camera_orbit_poses(
    n_views: int,
    distance: float = CAMERA_DIST,
    elevation_deg: float = 30.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate camera poses orbiting around the origin on a circle.

    Returns list of (R, t) pairs where R is 3x3 rotation and t is 3x1 translation
    such that world_to_camera = R @ p_world + t.
    """
    poses = []
    elev = np.deg2rad(elevation_deg)
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views

        # Camera position in world coordinates
        cam_x = distance * np.cos(elev) * np.cos(azimuth)
        cam_y = distance * np.cos(elev) * np.sin(azimuth)
        cam_z = distance * np.sin(elev)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Look-at matrix: camera looks at origin
        # forward = unit vector from camera toward origin (into the scene)
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-12)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-12

        # Rotation: world-to-camera (OpenCV convention: z forward into scene)
        # Camera axes: x=right, y=-up (image y points down), z=forward
        R = np.stack([right, -up, forward], axis=0)  # (3, 3)
        t = -R @ cam_pos  # (3,)

        poses.append((R, t.reshape(3, 1)))
    return poses


def perturb_pose(
    R: np.ndarray,
    t: np.ndarray,
    angle_sigma: float,
    trans_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add small perturbation to camera pose.

    angle_sigma: std of rotation perturbation in radians
    trans_sigma: std of translation perturbation (fraction of camera distance)
    """
    # Random rotation via axis-angle
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.normal(0.0, angle_sigma)
    # Rodrigues' rotation formula
    K_mat = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    dR = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
    R_new = dR @ R

    # Translation perturbation
    dt = rng.normal(0.0, trans_sigma * CAMERA_DIST, (3, 1))
    t_new = t + dt

    return R_new, t_new


# =============================================================================
# 3D Gaussian Representation
# =============================================================================

class Gaussian3D:
    """A single 3D Gaussian splat."""

    def __init__(
        self,
        mu: np.ndarray,       # (3,) centre
        scales: np.ndarray,   # (3,) axis-aligned scales (before rotation)
        rotation: np.ndarray, # (3, 3) rotation matrix
        opacity: float,       # alpha in [0, 1]
        color: np.ndarray,    # (3,) RGB in [0, 1]
    ):
        self.mu = mu.astype(np.float64)
        self.scales = scales.astype(np.float64)
        self.rotation = rotation.astype(np.float64)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.color = color.astype(np.float64).clip(0.0, 1.0)

        # Build 3x3 covariance: Sigma = R @ diag(s^2) @ R^T
        S = np.diag(self.scales ** 2)
        self.Sigma = self.rotation @ S @ self.rotation.T

    def project(
        self,
        R_cam: np.ndarray,
        t_cam: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Project this 3D Gaussian to a 2D Gaussian splat in image space.

        Returns (mu_2d, Sigma_2d, z_depth, projected_opacity).
        mu_2d is in pixel coordinates.
        """
        # Transform to camera coordinates
        mu_cam = R_cam @ self.mu.reshape(3, 1) + t_cam  # (3, 1)
        z = float(mu_cam[2, 0])
        if z <= 0.01:
            # Behind camera
            return np.array([0.0, 0.0]), np.eye(2), -1.0, 0.0

        # Project mean to pixel coords
        mu_proj = K @ mu_cam  # (3, 1)
        mu_2d = (mu_proj[:2, 0] / mu_proj[2, 0])  # (2,)

        # Jacobian of perspective projection at this depth
        fx, fy = K[0, 0], K[1, 1]
        J = np.array([
            [fx / z, 0, -fx * float(mu_cam[0, 0]) / (z * z)],
            [0, fy / z, -fy * float(mu_cam[1, 0]) / (z * z)],
        ], dtype=np.float64)

        # Project covariance to 2D
        Sigma_cam = R_cam @ self.Sigma @ R_cam.T
        Sigma_2d = J @ Sigma_cam @ J.T  # (2, 2)

        # Ensure positive definite by adding small diagonal
        Sigma_2d += np.eye(2) * 0.5

        return mu_2d, Sigma_2d, z, self.opacity


def eval_gaussian_2d(
    x_coords: np.ndarray,    # (H, W)
    y_coords: np.ndarray,    # (H, W)
    mu: np.ndarray,          # (2,)
    Sigma: np.ndarray,       # (2, 2)
) -> np.ndarray:
    """Evaluate a 2D Gaussian on a pixel grid. Returns (H, W) response."""
    dx = x_coords - mu[0]
    dy = y_coords - mu[1]

    # Inverse of Sigma
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    if abs(det) < 1e-12:
        return np.zeros_like(x_coords)
    inv_Sigma = np.array([
        [Sigma[1, 1], -Sigma[0, 1]],
        [-Sigma[1, 0], Sigma[0, 0]],
    ]) / det

    # Mahalanobis distance
    maha = (dx * (inv_Sigma[0, 0] * dx + inv_Sigma[0, 1] * dy) +
            dy * (inv_Sigma[1, 0] * dx + inv_Sigma[1, 1] * dy))

    # Clamp to avoid numerical overflow
    maha = np.clip(maha, 0.0, 50.0)
    return np.exp(-0.5 * maha)


# =============================================================================
# Rendering Engine (Alpha Compositing)
# =============================================================================

def render_gaussians(
    gaussians: list[Gaussian3D],
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    K: np.ndarray,
    H: int = IMAGE_SIZE,
    W: int = IMAGE_SIZE,
    bg_color: np.ndarray | None = None,
) -> np.ndarray:
    """Render a set of 3D Gaussians from a given camera pose.

    Uses front-to-back alpha compositing (depth-sorted).
    Optimised with per-Gaussian bounding-box culling (3-sigma radius).
    Returns (H, W, 3) float64 RGB image in [0, 1].
    """
    if bg_color is None:
        bg_color = np.array([0.0, 0.0, 0.0])

    # Project all Gaussians and sort by depth
    projections = []
    for g in gaussians:
        mu_2d, Sigma_2d, z, alpha = g.project(R_cam, t_cam, K)
        if z > 0 and alpha > 1e-4:
            projections.append((z, mu_2d, Sigma_2d, alpha, g.color))

    # Sort front-to-back (increasing z)
    projections.sort(key=lambda p: p[0])

    # Alpha compositing (front-to-back) with bounding-box optimisation
    image = np.zeros((H, W, 3), dtype=np.float64)
    T = np.ones((H, W), dtype=np.float64)  # accumulated transmittance

    for z, mu_2d, Sigma_2d, alpha, color in projections:
        # Compute 3-sigma bounding box from covariance eigenvalues
        sx = np.sqrt(max(Sigma_2d[0, 0], 0.5))
        sy = np.sqrt(max(Sigma_2d[1, 1], 0.5))
        radius = 3.0 * max(sx, sy)

        # Pixel-space bounding box
        x_lo = max(0, int(mu_2d[0] - radius))
        x_hi = min(W, int(mu_2d[0] + radius) + 1)
        y_lo = max(0, int(mu_2d[1] - radius))
        y_hi = min(H, int(mu_2d[1] + radius) + 1)

        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        # Evaluate 2D Gaussian only in the bounding box
        yy_loc, xx_loc = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        xx_f = xx_loc.astype(np.float64)
        yy_f = yy_loc.astype(np.float64)
        G_loc = eval_gaussian_2d(xx_f, yy_f, mu_2d, Sigma_2d)

        # Weight = opacity * Gaussian response
        weight = alpha * G_loc
        T_loc = T[y_lo:y_hi, x_lo:x_hi]

        # Accumulate colour: C += T * weight * color
        tw = T_loc * weight
        for c in range(3):
            image[y_lo:y_hi, x_lo:x_hi, c] += tw * color[c]

        # Update transmittance
        T[y_lo:y_hi, x_lo:x_hi] = T_loc * (1.0 - weight)

    T = np.clip(T, 0.0, 1.0)
    # Background
    for c in range(3):
        image[:, :, c] += T * bg_color[c]

    return np.clip(image, 0.0, 1.0)


# =============================================================================
# Scene Generators (3D Phantoms)
# =============================================================================

def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly random 3x3 rotation matrix."""
    # QR decomposition of random matrix
    A = rng.standard_normal((3, 3))
    Q, R_diag = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    Q *= np.sign(np.diag(R_diag))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def generate_sphere_cluster(
    rng: np.random.Generator,
    n_spheres: int = 5,
    complexity: str = "simple",
) -> list[Gaussian3D]:
    """Generate a cluster of coloured spheres as 3D Gaussians.

    Each sphere is represented by several concentric Gaussians of varying
    scale to approximate a smooth sphere surface.
    """
    gaussians = []

    if complexity == "simple":
        n_spheres = int(rng.integers(3, 7))
        n_shells = 2
    elif complexity == "medium":
        n_spheres = int(rng.integers(5, 12))
        n_shells = 3
    else:
        n_spheres = int(rng.integers(10, 25))
        n_shells = 4

    for _ in range(n_spheres):
        centre = rng.uniform(-1.0, 1.0, 3)
        radius = float(rng.uniform(0.15, 0.6))
        base_color = rng.uniform(0.2, 1.0, 3)
        opacity = float(rng.uniform(0.5, 0.95))

        for s in range(n_shells):
            scale_factor = radius * (0.5 + 0.5 * s / max(n_shells - 1, 1))
            scales = np.array([scale_factor] * 3) * float(rng.uniform(0.8, 1.2))
            rot = random_rotation_matrix(rng)
            shell_opacity = opacity * (1.0 - 0.2 * s / max(n_shells - 1, 1))
            color_var = base_color * float(rng.uniform(0.85, 1.15))
            gaussians.append(Gaussian3D(
                mu=centre,
                scales=scales,
                rotation=rot,
                opacity=shell_opacity,
                color=np.clip(color_var, 0.0, 1.0),
            ))

    return gaussians


def generate_geometric_scene(
    rng: np.random.Generator,
    complexity: str = "simple",
) -> list[Gaussian3D]:
    """Generate a scene with geometric primitives (boxes, ellipsoids, cylinders)
    approximated by anisotropic 3D Gaussians.
    """
    gaussians = []

    if complexity == "simple":
        n_objects = int(rng.integers(3, 8))
    elif complexity == "medium":
        n_objects = int(rng.integers(6, 15))
    else:
        n_objects = int(rng.integers(12, 30))

    for _ in range(n_objects):
        centre = rng.uniform(-1.2, 1.2, 3)
        obj_type = rng.choice(["ellipsoid", "elongated", "flat", "sphere"])

        if obj_type == "ellipsoid":
            scales = rng.uniform(0.1, 0.5, 3)
        elif obj_type == "elongated":
            # One axis much longer (cylinder-like)
            short = float(rng.uniform(0.08, 0.2))
            long_s = float(rng.uniform(0.4, 0.8))
            scales = np.array([short, short, long_s])
            rng.shuffle(scales)
        elif obj_type == "flat":
            # One axis much shorter (disc-like)
            wide = float(rng.uniform(0.3, 0.6))
            thin = float(rng.uniform(0.03, 0.1))
            scales = np.array([wide, wide, thin])
            rng.shuffle(scales)
        else:
            r = float(rng.uniform(0.15, 0.4))
            scales = np.array([r, r, r])

        rot = random_rotation_matrix(rng)
        opacity = float(rng.uniform(0.4, 0.95))
        color = rng.uniform(0.1, 1.0, 3)

        gaussians.append(Gaussian3D(
            mu=centre,
            scales=scales,
            rotation=rot,
            opacity=opacity,
            color=color,
        ))

        # Add secondary detail Gaussians on surfaces
        if rng.random() > 0.4:
            n_detail = int(rng.integers(1, 4))
            for _ in range(n_detail):
                offset = rng.standard_normal(3) * scales * 0.8
                detail_centre = centre + offset
                detail_scales = scales * float(rng.uniform(0.2, 0.5))
                detail_color = np.clip(color + rng.uniform(-0.2, 0.2, 3), 0.0, 1.0)
                gaussians.append(Gaussian3D(
                    mu=detail_centre,
                    scales=detail_scales,
                    rotation=random_rotation_matrix(rng),
                    opacity=float(rng.uniform(0.3, 0.8)),
                    color=detail_color,
                ))

    return gaussians


def generate_overlapping_ellipsoids(
    rng: np.random.Generator,
    complexity: str = "simple",
) -> list[Gaussian3D]:
    """Generate overlapping translucent ellipsoids -- tests alpha compositing."""
    gaussians = []

    if complexity == "simple":
        n_ellipsoids = int(rng.integers(4, 8))
    elif complexity == "medium":
        n_ellipsoids = int(rng.integers(8, 16))
    else:
        n_ellipsoids = int(rng.integers(15, 30))

    # Create overlapping clusters
    n_clusters = max(1, n_ellipsoids // 4)
    cluster_centres = rng.uniform(-0.6, 0.6, (n_clusters, 3))

    for i in range(n_ellipsoids):
        cluster = i % n_clusters
        offset = rng.standard_normal(3) * 0.4
        centre = cluster_centres[cluster] + offset

        scales = rng.uniform(0.15, 0.55, 3)
        rot = random_rotation_matrix(rng)
        # Lower opacity for overlap visibility
        opacity = float(rng.uniform(0.25, 0.7))
        # Distinct colours per cluster
        hue_offset = cluster / max(n_clusters, 1) * 2 * np.pi
        color = np.array([
            0.5 + 0.4 * np.cos(hue_offset),
            0.5 + 0.4 * np.cos(hue_offset + 2 * np.pi / 3),
            0.5 + 0.4 * np.cos(hue_offset + 4 * np.pi / 3),
        ]) + rng.uniform(-0.15, 0.15, 3)

        gaussians.append(Gaussian3D(
            mu=centre,
            scales=scales,
            rotation=rot,
            opacity=opacity,
            color=np.clip(color, 0.0, 1.0),
        ))

    return gaussians


def generate_scattered_points(
    rng: np.random.Generator,
    complexity: str = "simple",
) -> list[Gaussian3D]:
    """Generate a point-cloud-like scene with many small Gaussians."""
    if complexity == "simple":
        n_points = int(rng.integers(20, 50))
    elif complexity == "medium":
        n_points = int(rng.integers(40, 80))
    else:
        n_points = int(rng.integers(60, 120))

    gaussians = []
    # Generate a 3D structure (e.g., torus, helix, surface)
    structure = rng.choice(["torus", "helix", "random_surface", "cloud"])

    for i in range(n_points):
        t_param = float(i) / n_points * 2 * np.pi

        if structure == "torus":
            R_major = 0.8
            R_minor = 0.25
            phi = t_param
            theta = float(rng.uniform(0, 2 * np.pi))
            x = (R_major + R_minor * np.cos(theta)) * np.cos(phi)
            y = (R_major + R_minor * np.cos(theta)) * np.sin(phi)
            z = R_minor * np.sin(theta)
            centre = np.array([x, y, z]) + rng.standard_normal(3) * 0.05
        elif structure == "helix":
            x = 0.6 * np.cos(t_param * 3)
            y = 0.6 * np.sin(t_param * 3)
            z = -1.0 + 2.0 * float(i) / n_points
            centre = np.array([x, y, z]) + rng.standard_normal(3) * 0.05
        elif structure == "random_surface":
            u = float(rng.uniform(-1, 1))
            v = float(rng.uniform(-1, 1))
            z = 0.3 * np.sin(2 * u * np.pi) * np.cos(2 * v * np.pi)
            centre = np.array([u, v, z]) + rng.standard_normal(3) * 0.03
        else:
            centre = rng.standard_normal(3) * 0.5

        scale = float(rng.uniform(0.03, 0.12))
        scales = np.array([scale, scale, scale]) * rng.uniform(0.6, 1.4, 3)

        # Colour based on position
        color = np.array([
            0.5 + 0.5 * np.sin(t_param),
            0.5 + 0.5 * np.sin(t_param + 2),
            0.5 + 0.5 * np.sin(t_param + 4),
        ])

        gaussians.append(Gaussian3D(
            mu=centre,
            scales=scales,
            rotation=random_rotation_matrix(rng),
            opacity=float(rng.uniform(0.5, 0.95)),
            color=np.clip(color, 0.0, 1.0),
        ))

    return gaussians


def generate_nested_structure(
    rng: np.random.Generator,
    complexity: str = "simple",
) -> list[Gaussian3D]:
    """Generate nested concentric structures -- tests depth ordering."""
    gaussians = []

    if complexity == "simple":
        n_layers = int(rng.integers(3, 5))
    elif complexity == "medium":
        n_layers = int(rng.integers(4, 7))
    else:
        n_layers = int(rng.integers(6, 10))

    for layer in range(n_layers):
        r = 0.2 + 0.15 * layer
        n_per_layer = 4 + 2 * layer
        opacity = 0.3 + 0.1 * (n_layers - layer) / n_layers

        # Colour gradient from inner to outer
        hue = float(layer) / n_layers
        base_color = np.array([
            0.3 + 0.6 * hue,
            0.3 + 0.6 * (1 - hue),
            0.5,
        ])

        for j in range(n_per_layer):
            phi = 2 * np.pi * j / n_per_layer + rng.uniform(-0.1, 0.1)
            theta = np.pi / 2 + rng.uniform(-0.3, 0.3)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            centre = np.array([x, y, z])

            scale = float(rng.uniform(0.05, 0.15))
            scales = np.array([scale, scale, scale * float(rng.uniform(0.5, 2.0))])

            gaussians.append(Gaussian3D(
                mu=centre,
                scales=scales,
                rotation=random_rotation_matrix(rng),
                opacity=float(np.clip(opacity + rng.uniform(-0.1, 0.1), 0.2, 0.9)),
                color=np.clip(base_color + rng.uniform(-0.15, 0.15, 3), 0.0, 1.0),
            ))

    return gaussians


# =============================================================================
# Phantom Dispatcher
# =============================================================================

PHANTOM_TYPES = {
    "sphere_cluster_simple":     lambda rng: generate_sphere_cluster(rng, complexity="simple"),
    "sphere_cluster_medium":     lambda rng: generate_sphere_cluster(rng, complexity="medium"),
    "sphere_cluster_complex":    lambda rng: generate_sphere_cluster(rng, complexity="complex"),
    "geometric_simple":          lambda rng: generate_geometric_scene(rng, complexity="simple"),
    "geometric_medium":          lambda rng: generate_geometric_scene(rng, complexity="medium"),
    "geometric_complex":         lambda rng: generate_geometric_scene(rng, complexity="complex"),
    "overlapping_simple":        lambda rng: generate_overlapping_ellipsoids(rng, complexity="simple"),
    "overlapping_medium":        lambda rng: generate_overlapping_ellipsoids(rng, complexity="medium"),
    "overlapping_complex":       lambda rng: generate_overlapping_ellipsoids(rng, complexity="complex"),
    "scattered_simple":          lambda rng: generate_scattered_points(rng, complexity="simple"),
    "scattered_medium":          lambda rng: generate_scattered_points(rng, complexity="medium"),
    "scattered_complex":         lambda rng: generate_scattered_points(rng, complexity="complex"),
    "nested_simple":             lambda rng: generate_nested_structure(rng, complexity="simple"),
    "nested_medium":             lambda rng: generate_nested_structure(rng, complexity="medium"),
    "nested_complex":            lambda rng: generate_nested_structure(rng, complexity="complex"),
}

TIER_PHANTOMS = {
    "public": [
        "sphere_cluster_simple", "geometric_simple", "overlapping_simple",
        "scattered_simple", "nested_simple", "sphere_cluster_medium",
        "geometric_medium", "overlapping_medium", "scattered_medium",
        "nested_medium", "sphere_cluster_simple", "geometric_simple",
    ],
    "dev": [
        "sphere_cluster_simple", "sphere_cluster_medium", "sphere_cluster_complex",
        "geometric_simple", "geometric_medium", "geometric_complex",
        "overlapping_simple", "overlapping_medium", "overlapping_complex",
        "scattered_simple", "scattered_medium", "scattered_complex",
        "nested_simple", "nested_medium", "nested_complex",
        "sphere_cluster_medium", "geometric_medium", "overlapping_medium",
        "scattered_medium", "nested_medium",
    ],
    "hidden": [
        "sphere_cluster_complex", "sphere_cluster_complex", "geometric_complex",
        "geometric_complex", "overlapping_complex", "overlapping_complex",
        "scattered_complex", "scattered_complex", "nested_complex",
        "nested_complex", "sphere_cluster_complex", "geometric_complex",
        "overlapping_complex", "scattered_complex", "nested_complex",
        "sphere_cluster_complex", "geometric_complex", "overlapping_complex",
        "scattered_complex", "nested_complex",
    ],
}

TIER_CONFIG = {
    "public": {"n_samples": 12, "base_seed": 0,     "n_views": 8},
    "dev":    {"n_samples": 20, "base_seed": 10000,  "n_views": 8},
    "hidden": {"n_samples": 20, "base_seed": 20000,  "n_views": 8},
}


# =============================================================================
# Forward Model: Multi-View Gaussian Splatting
# =============================================================================

def forward_model(
    gaussians: list[Gaussian3D],
    n_views: int = N_VIEWS_DEFAULT,
    elevation_deg: float = 30.0,
    bg_color: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Render ground truth multi-view images.

    Returns
    -------
    x_true : (H, W, 3) float64 -- reference view (view 0) RGB image
    views_data : list of dicts with {image, R, t, K} for all views
    H_ideal : (n_views, 3, 4) camera projection matrices [K @ [R|t]]
    """
    if bg_color is None:
        bg_color = np.array([0.0, 0.0, 0.0])

    K = make_intrinsic(FOCAL_LENGTH, IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0)
    poses = camera_orbit_poses(n_views, elevation_deg=elevation_deg)

    views_data = []
    images = []
    H_ideal = np.zeros((n_views, 3, 4), dtype=np.float64)

    for v, (R, t) in enumerate(poses):
        img = render_gaussians(gaussians, R, t, K, IMAGE_SIZE, IMAGE_SIZE, bg_color)
        images.append(img)
        views_data.append({"image": img, "R": R, "t": t, "K": K})
        # Camera projection matrix P = K @ [R | t]
        Rt = np.hstack([R, t])  # (3, 4)
        H_ideal[v] = K @ Rt

    # Reference view is view 0
    x_true = images[0]
    return x_true, views_data, H_ideal


def apply_mismatch(
    gaussians: list[Gaussian3D],
    views_data: list[dict],
    mismatch: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mismatch to the Gaussian splatting measurement.

    1. Camera pose error: perturb camera extrinsics
    2. Depth uncertainty: perturb Gaussian z-positions
    3. Opacity noise: add noise to opacity values
    4. View sparsity: drop some views (replace with noise)

    Returns
    -------
    y : (V, H, W, 3) float32 -- noisy multi-view measurements
    H_ideal : (V, 3, 4) float64 -- nominal (mismatched) camera matrices
    """
    n_views = len(views_data)
    K = views_data[0]["K"]
    bg_color = np.array([0.0, 0.0, 0.0])

    pose_err = mismatch["camera_pose_error"]
    depth_err = mismatch["depth_uncertainty"]
    opacity_sigma = mismatch["opacity_noise"]
    sparsity = mismatch["view_sparsity"]

    # Determine which views to drop
    n_drop = int(round(sparsity * n_views))
    # Always keep view 0 (reference)
    droppable = list(range(1, n_views))
    rng.shuffle(droppable)
    dropped_views = set(droppable[:n_drop])

    # Create perturbed Gaussians (depth + opacity noise)
    perturbed_gs = []
    for g in gaussians:
        # Depth uncertainty: perturb z-coordinate
        new_mu = g.mu.copy()
        new_mu[2] += rng.normal(0.0, depth_err * SCENE_RADIUS)

        # Opacity noise
        new_opacity = g.opacity + rng.normal(0.0, opacity_sigma)
        new_opacity = float(np.clip(new_opacity, 0.0, 1.0))

        perturbed_gs.append(Gaussian3D(
            mu=new_mu,
            scales=g.scales,
            rotation=g.rotation,
            opacity=new_opacity,
            color=g.color,
        ))

    # Render with perturbed cameras and perturbed Gaussians
    y_views = np.zeros((n_views, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    H_noisy = np.zeros((n_views, 3, 4), dtype=np.float64)

    for v in range(n_views):
        R_true = views_data[v]["R"]
        t_true = views_data[v]["t"]

        # Perturb camera pose
        R_noisy, t_noisy = perturb_pose(R_true, t_true, pose_err, pose_err, rng)

        if v in dropped_views:
            # Dropped view: fill with noise (simulating missing data)
            noise_level = float(rng.uniform(0.1, 0.3))
            y_views[v] = rng.uniform(0.0, noise_level,
                                     (IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
        else:
            # Render with perturbed parameters
            img = render_gaussians(
                perturbed_gs, R_noisy, t_noisy, K,
                IMAGE_SIZE, IMAGE_SIZE, bg_color,
            )
            # Add measurement noise (sensor noise)
            sensor_noise = rng.normal(0.0, 0.02, img.shape)
            y_views[v] = np.clip(img + sensor_noise, 0.0, 1.0).astype(np.float32)

        # Store nominal (mismatched) projection matrix
        Rt_noisy = np.hstack([R_noisy, t_noisy])
        H_noisy[v] = K @ Rt_noisy

    return y_views, H_noisy


# =============================================================================
# Baseline Reconstruction
# =============================================================================

def baseline_reconstruct(
    y: np.ndarray,
    H_ideal: np.ndarray,
) -> np.ndarray:
    """CPU baseline: simple averaging of available views after de-projection.

    This is the naive baseline that does NOT correct for mismatch.
    Simply averages all non-dropped views (heuristic: drop views with
    very low variance, which are likely noise-filled dropped views).

    Returns (H, W, 3) float64 reconstruction.
    """
    n_views = y.shape[0]

    # Detect dropped views by variance (dropped views have low information)
    view_vars = []
    for v in range(n_views):
        view_vars.append(float(np.var(y[v])))
    median_var = float(np.median(view_vars))

    # Average valid views
    accum = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float64)
    count = 0
    for v in range(n_views):
        if view_vars[v] > median_var * 0.3:  # not obviously dropped
            accum += y[v].astype(np.float64)
            count += 1

    if count == 0:
        return y[0].astype(np.float64)

    recon = accum / count

    # Light smoothing to reduce noise
    for c in range(3):
        recon[:, :, c] = gaussian_filter(recon[:, :, c], sigma=0.5)

    return np.clip(recon, 0.0, 1.0)


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """PSNR on RGB images, data range [0, 1]."""
    x_t = x_true.astype(np.float64)
    x_r = x_recon.astype(np.float64)
    mse = np.mean((x_t - x_r) ** 2)
    if mse < 1e-15:
        return 60.0
    return float(10 * np.log10(1.0 / mse))


def compute_ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified SSIM on RGB images (per-channel mean)."""
    ssim_sum = 0.0
    n_ch = min(x.shape[2], y.shape[2]) if x.ndim == 3 else 1

    for c in range(n_ch):
        if x.ndim == 3:
            a = x[:, :, c].astype(np.float64)
            b = y[:, :, c].astype(np.float64)
        else:
            a = x.astype(np.float64)
            b = y.astype(np.float64)

        mu_a = a.mean()
        mu_b = b.mean()
        sig_a = a.std()
        sig_b = b.std()
        sig_ab = np.mean((a - mu_a) * (b - mu_b))

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_val = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
                   ((mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2))
        ssim_sum += ssim_val

    return float(ssim_sum / max(n_ch, 1))


# =============================================================================
# Image Utilities
# =============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _save_png(arr: np.ndarray, path: Path) -> None:
    """Save array as PNG. Handles both grayscale and RGB."""
    normed = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    if normed.ndim == 3 and normed.shape[2] == 3:
        Image.fromarray(normed, "RGB").save(str(path))
    else:
        Image.fromarray(normed, "L").save(str(path))


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale for single-channel display."""
    if img.ndim == 3 and img.shape[2] == 3:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img


# =============================================================================
# Dataset Tier Generator
# =============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(tier: str) -> dict:
    """Generate one tier of the Gaussian splatting benchmark dataset.

    Returns dict of per-sample baseline metrics.
    """
    config = TIER_CONFIG[tier]
    spec_ranges = SPEC[tier]
    n_samples = config["n_samples"]
    base_seed = config["base_seed"]
    n_views = config["n_views"]
    phantom_types = TIER_PHANTOMS[tier]

    tier_dir = BENCHMARK_DIR / tier
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    tier_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"gaussian_splatting_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)

    results = {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM 3D Gaussian Splatting benchmark -- {tier} tier "
            f"(multi-view rendering, alpha compositing)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["physics"] = json.dumps({
            "forward_model": (
                "y = sum_k(T_k * alpha_k * G_2d(mu_k, Sigma_k) * c_k) + noise, "
                "where G_2d is 2D Gaussian from projecting 3D Gaussian by camera"
            ),
            "n_views": n_views,
            "image_size": IMAGE_SIZE,
            "focal_length_px": FOCAL_LENGTH,
            "camera_distance": CAMERA_DIST,
            "scene_radius": SCENE_RADIUS,
            "measurement_type": "sparse_multiview_rgb",
        })

        for idx in range(n_samples):
            key = f"sample_{idx:02d}"
            sample_seed = base_seed + idx
            sample_rng = np.random.default_rng(sample_seed)

            # Generate 3D scene
            phantom_type = phantom_types[idx % len(phantom_types)]
            print(f"  [{tier}] {key} generating {phantom_type}...")
            scene_gaussians = PHANTOM_TYPES[phantom_type](sample_rng)

            # Vary elevation slightly per sample
            elev = 30.0 + float(rng.uniform(-10, 10))

            # Render ground truth views
            x_true, views_data, H_ideal_clean = forward_model(
                scene_gaussians, n_views=n_views, elevation_deg=elev,
            )

            # Sample mismatch parameters
            mis = sample_mismatch(rng, spec_ranges)

            # Apply mismatch: render with perturbed cameras/Gaussians + noise
            y, H_ideal = apply_mismatch(
                scene_gaussians, views_data, mis, rng,
            )

            # CPU baseline: average available views
            recon = baseline_reconstruct(y, H_ideal)

            # Metrics (compare reference view reconstruction)
            psnr_val = compute_psnr(x_true, recon)
            ssim_val = compute_ssim_simple(x_true, recon)

            # Store in HDF5
            grp = f.create_group(key)

            # x_true: (H, W, 3) reference RGB image
            grp.create_dataset(
                "x_true", data=x_true.astype(np.float32),
                compression="gzip",
            )

            # y: (n_views, H, W, 3) multi-view measurements
            grp.create_dataset(
                "y", data=y, compression="gzip",
            )

            # H_ideal: (n_views, 3, 4) camera projection matrices
            grp.create_dataset(
                "H_ideal", data=H_ideal.astype(np.float32),
                compression="gzip",
            )

            grp.attrs["metadata"] = json.dumps({
                "phantom_type": phantom_type,
                "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
                "n_views": n_views,
                "n_gaussians": len(scene_gaussians),
                "seed": sample_seed,
                "elevation_deg": round(elev, 2),
                "baseline_psnr_db": round(psnr_val, 2),
                "baseline_ssim": round(ssim_val, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps(mis)

            results[key] = {
                "phantom_type": phantom_type,
                "n_gaussians": len(scene_gaussians),
                "psnr_db": round(psnr_val, 2),
                "ssim": round(ssim_val, 4),
                "mismatch": mis,
            }

            print(f"  [{tier}] {key} {phantom_type:30s} "
                  f"#G={len(scene_gaussians):3d}  "
                  f"PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}  "
                  f"pose_err={mis['camera_pose_error']:.4f} "
                  f"depth_err={mis['depth_uncertainty']:.4f} "
                  f"opacity_noise={mis['opacity_noise']:.4f} "
                  f"sparsity={mis['view_sparsity']:.2f}")

    # Save spec
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "results.json", "w") as rf:
        json.dump(results, rf, indent=2)

    print(f"  [{tier}] HDF5 -> {h5_path.name}  ({n_samples} samples)")
    return results


# =============================================================================
# Gallery Image Generation
# =============================================================================

def generate_gallery(n_scenes: int = 4) -> None:
    """Generate gallery preview images for the platform.

    Creates scene_XX directories with:
        gt.png              -- ground truth reference view
        measurement_I.png   -- multi-view obs (low noise, view 0)
        measurement_II.png  -- multi-view obs (high noise, view 0)
        recon_I.png         -- baseline reconstruction (low noise)
        recon_II.png        -- baseline reconstruction (high noise)
        recon_III.png       -- view from different angle
    """
    gallery_root = (
        Path(__file__).resolve().parents[3]
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "gaussian_splatting"
    )

    print(f"\nGenerating gallery images -> {gallery_root}")

    phantom_gallery = [
        "sphere_cluster_medium", "geometric_medium",
        "overlapping_medium", "scattered_medium",
    ]

    for scene_idx in range(n_scenes):
        scene_dir = gallery_root / f"scene_{scene_idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42000 + scene_idx)
        ptype = phantom_gallery[scene_idx % len(phantom_gallery)]
        scene_gs = PHANTOM_TYPES[ptype](rng)

        # Ground truth: reference view
        x_true, views_data, H_clean = forward_model(scene_gs, n_views=8)
        _save_png(x_true, scene_dir / "gt.png")

        # Measurement I: low noise
        mis_low = {
            "camera_pose_error": 0.01,
            "depth_uncertainty": 0.01,
            "opacity_noise": 0.02,
            "view_sparsity": 0.0,
        }
        y_low, _ = apply_mismatch(scene_gs, views_data, mis_low, rng)
        _save_png(y_low[0], scene_dir / "measurement_I.png")

        # Measurement II: high noise
        mis_high = {
            "camera_pose_error": 0.08,
            "depth_uncertainty": 0.08,
            "opacity_noise": 0.12,
            "view_sparsity": 0.3,
        }
        y_high, H_high = apply_mismatch(scene_gs, views_data, mis_high, rng)
        _save_png(y_high[0], scene_dir / "measurement_II.png")

        # Reconstruction I: baseline on low noise
        recon_low = baseline_reconstruct(y_low, H_clean)
        _save_png(recon_low, scene_dir / "recon_I.png")

        # Reconstruction II: baseline on high noise
        recon_high = baseline_reconstruct(y_high, H_high)
        _save_png(recon_high, scene_dir / "recon_II.png")

        # Reconstruction III: different view angle
        _save_png(y_low[3], scene_dir / "recon_III.png")

        print(f"  scene_{scene_idx:02d}: {ptype} -> 6 images")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("3D Gaussian Splatting Benchmark Dataset Generator")
    print("=" * 65)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Focal length: {FOCAL_LENGTH} px")
    print(f"Camera distance: {CAMERA_DIST}")
    print(f"Scene radius: {SCENE_RADIUS}")
    print()

    all_results = {}

    for tier in ["public", "dev", "hidden"]:
        n = TIER_CONFIG[tier]["n_samples"]
        print(f"Generating {tier} tier ({n} samples)...")
        results = generate_tier(tier)
        all_results[tier] = results

        # Summarize
        psnrs = [v["psnr_db"] for v in results.values()]
        ssims = [v["ssim"] for v in results.values()]
        print(f"  [{tier}] Baseline PSNR: {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB")
        print(f"  [{tier}] Baseline SSIM: {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")
        print()

    # Generate gallery images
    generate_gallery(n_scenes=4)

    # Save overall summary
    summary_path = BENCHMARK_DIR / "baseline_summary.json"
    with open(summary_path, "w") as sf:
        json.dump(all_results, sf, indent=2)
    print(f"\nBaseline summary -> {summary_path}")

    print(f"\n{'=' * 65}")
    print("Done -- 3D Gaussian Splatting benchmark ready")
    print(f"  public:  {TIER_CONFIG['public']['n_samples']} samples")
    print(f"  dev:     {TIER_CONFIG['dev']['n_samples']} samples")
    print(f"  hidden:  {TIER_CONFIG['hidden']['n_samples']} samples")

    for tier in ["public", "dev", "hidden"]:
        psnrs = [v["psnr_db"] for v in all_results[tier].values()]
        ssims = [v["ssim"] for v in all_results[tier].values()]
        print(f"  {tier:8s} avg PSNR={np.mean(psnrs):.2f} dB  avg SSIM={np.mean(ssims):.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
