import torch
import torch.nn as nn

def chamfer_distance_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """
    Computes bidirectional Chamfer Distance Loss between predicted point cloud and target point cloud.
    pred_points: (Batch, N, 3) or (N, 3)
    target_points: (Batch, M, 3) or (M, 3)
    """
    if pred_points.dim() == 2:
        pred_points = pred_points.unsqueeze(0)
    if target_points.dim() == 2:
        target_points = target_points.unsqueeze(0)

    # Compute pairwise squared Euclidean distances (Batch, N, M)
    dist_matrix = torch.cdist(pred_points, target_points, p=2) ** 2

    # Minimum distance from each predicted point to nearest target point
    min_dist_pred = torch.min(dist_matrix, dim=2)[0]
    
    # Minimum distance from each target point to nearest predicted point
    min_dist_target = torch.min(dist_matrix, dim=1)[0]

    # Bidirectional Chamfer Loss
    loss = torch.mean(min_dist_pred) + torch.mean(min_dist_target)
    return loss

class ChamferLoss(nn.Module):
    def __init__(self, max_points: int = 1024):
        super(ChamferLoss, self).__init__()
        self.max_points = max_points

    def forward(self, pred, target):
        # If input is (B, H, W), unsqueeze to (B, 1, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        if pred.dim() == 4:
            # Reconstruct 3D points from depth maps
            B, _, H, W = pred.shape
            device = pred.device
            
            # Downsample if too large to avoid OOM
            if H * W > self.max_points:
                scale = int(round((H * W / self.max_points) ** 0.5))
                if scale > 1:
                    pred = torch.nn.functional.avg_pool2d(pred, kernel_size=scale, stride=scale)
                    target = torch.nn.functional.avg_pool2d(target, kernel_size=scale, stride=scale)
            
            # Reconstruct 3D coordinates
            B_sub, _, H_sub, W_sub = pred.shape
            
            # Create grid (matches dataset projection)
            # grid_x goes from -0.5 to 0.5 (W columns)
            # grid_y goes from 0.5 to -0.5 (H rows)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.5, -0.5, H_sub, device=device),
                torch.linspace(-0.5, 0.5, W_sub, device=device),
                indexing='ij'
            )
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B_sub, -1, -1, -1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B_sub, -1, -1, -1)
            grid_z = pred - 0.5
            pred_pts = torch.cat([grid_x, grid_y, grid_z], dim=1).view(B_sub, 3, -1).permute(0, 2, 1)
            
            grid_z_target = target - 0.5
            target_pts = torch.cat([grid_x, grid_y, grid_z_target], dim=1).view(B_sub, 3, -1).permute(0, 2, 1)
            
            return chamfer_distance_loss(pred_pts, target_pts)
            
        return chamfer_distance_loss(pred, target)


class LaplacianEdgeLoss(nn.Module):
    """
    Laplacian mesh edge regularization loss.
    Penalizes deviation of each vertex from the average of its neighbors.
    L = sum_i || v_i - (1/|N(i)|) * sum_{j in N(i)} v_j ||^2

    Prevents vertex collapse and preserves sharp features during training.
    Works on 2D feature maps by treating each pixel as a vertex with 4-connected neighbors.
    """
    def __init__(self):
        super(LaplacianEdgeLoss, self).__init__()
        # Laplacian kernel for 2D convolution (4-connected neighborhood)
        kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0, 1.0, 0.0]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred: (Batch, 1, H, W) predicted depth/height map
        Returns: scalar Laplacian smoothness loss
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        # Apply Laplacian convolution
        laplacian = torch.nn.functional.conv2d(pred, self.kernel.to(pred.device), padding=1)
        loss = torch.mean(laplacian ** 2)
        return loss


class NormalConsistencyLoss(nn.Module):
    """
    Normal consistency loss computed from predicted depth/height maps.
    Computes surface normals from depth gradients and penalizes inconsistency
    between neighboring normal vectors.

    L = 1 - mean(dot(n_i, n_j)) for adjacent pixels i, j

    Encourages smooth surfaces while preserving deliberate sharp edges.
    """
    def __init__(self):
        super(NormalConsistencyLoss, self).__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred: (Batch, 1, H, W) predicted depth/height map
        Returns: scalar normal consistency loss
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        # Compute depth gradients (finite differences)
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # (B, 1, H, W-1)
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # (B, 1, H-1, W)

        # Trim to common size
        min_h = min(dx.shape[2], dy.shape[2])
        min_w = min(dx.shape[3], dy.shape[3])
        dx = dx[:, :, :min_h, :min_w]
        dy = dy[:, :, :min_h, :min_w]

        # Construct normal vectors: (-dx, -dy, 1.0)
        ones = torch.ones_like(dx)
        normals = torch.cat([-dx, -dy, ones], dim=1)  # (B, 3, H', W')

        # Normalize
        norm_mag = torch.norm(normals, dim=1, keepdim=True) + 1e-8
        normals = normals / norm_mag

        # Compute consistency: dot product between horizontally adjacent normals
        dot_h = torch.sum(normals[:, :, :, :-1] * normals[:, :, :, 1:], dim=1)
        # Dot product between vertically adjacent normals
        dot_v = torch.sum(normals[:, :, :-1, :] * normals[:, :, 1:, :], dim=1)

        # Loss = 1 - mean(dot products) — perfect consistency would yield 0
        loss = 1.0 - 0.5 * (torch.mean(dot_h) + torch.mean(dot_v))
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss for depth map prediction.
    
    Unlike pixel-wise MSE which treats every pixel independently, SSIM captures
    local structural patterns (luminance, contrast, structure) that are critical
    for accurate 3D surface reconstruction. L = 1 - SSIM.
    """
    def __init__(self, window_size: int = 11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        mu_p = torch.nn.functional.avg_pool2d(
            pred, self.window_size, stride=1, padding=self.padding)
        mu_t = torch.nn.functional.avg_pool2d(
            target, self.window_size, stride=1, padding=self.padding)

        sigma_p_sq = torch.nn.functional.avg_pool2d(
            pred ** 2, self.window_size, stride=1, padding=self.padding) - mu_p ** 2
        sigma_t_sq = torch.nn.functional.avg_pool2d(
            target ** 2, self.window_size, stride=1, padding=self.padding) - mu_t ** 2
        sigma_pt = torch.nn.functional.avg_pool2d(
            pred * target, self.window_size, stride=1, padding=self.padding) - mu_p * mu_t

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2))

        return 1.0 - ssim_map.mean()


class MaskedMSELoss(nn.Module):
    """
    Foreground-masked MSE loss.
    
    Only penalises prediction error on pixels where the ground-truth depth is
    non-zero (i.e. the object), so background pixels don't dilute the gradient
    signal. Falls back to standard MSE if no foreground pixels exist.
    """
    def __init__(self, threshold: float = 0.01):
        super(MaskedMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_mask = (target > self.threshold).float()
        fg_count = fg_mask.sum()
        if fg_count < 1.0:
            return torch.nn.functional.mse_loss(pred, target)
        masked_sq_err = fg_mask * (pred - target) ** 2
        return masked_sq_err.sum() / fg_count

