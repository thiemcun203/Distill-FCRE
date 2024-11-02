import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
import numpy as np
from typing import Optional, Union, Literal

def min_max_scaling(C):
    """Min-max scaling for stabilization"""
    eps = 1e-10
    if isinstance(C, torch.Tensor):
        C_min = torch.min(C)
        C_max = torch.max(C)
    else:
        C_min = np.min(C)
        C_max = np.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C

def compute_distance_matrix_cosine(x1, x2):
    """Compute cosine distance matrix between two sets of vectors"""
    x1_norm = F.normalize(x1, p=2, dim=-1)
    x2_norm = F.normalize(x2, p=2, dim=-1)
    C = torch.matmul(x1_norm, x2_norm.t())
    C = (C + 1.0) / 2.0
    C = 1.0 - C
    return C

def compute_distance_matrix_l2(x1, x2):
    """Compute L2 distance matrix between two sets of vectors"""
    C = torch.cdist(x1, x2, p=2)
    C = min_max_scaling(C)
    return C

def compute_weights_uniform(x1, x2):
    """Compute uniform weights for the vectors"""
    n1 = x1.size(0)
    n2 = x2.size(0)
    weights1 = torch.ones(n1, device=x1.device, dtype=x1.dtype) / n1
    weights2 = torch.ones(n2, device=x2.device, dtype=x2.dtype) / n2
    return weights1, weights2

def compute_weights_norm(s1_embeddings, s2_embeddings):
    """Compute weights based on L2 norm of embeddings"""
    s1_weights = torch.norm(s1_embeddings, p=2, dim=-1)
    s2_weights = torch.norm(s2_embeddings, p=2, dim=-1)
    # Normalize to sum to 1
    s1_weights = s1_weights / s1_weights.sum()
    s2_weights = s2_weights / s2_weights.sum()
    return s1_weights, s2_weights

def to_numpy_safe(tensor):
    """Safely convert tensor to numpy array, handling bfloat16 case"""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().detach().cpu().numpy()
    return tensor.detach().cpu().numpy()

class OTLoss(nn.Module):
    def __init__(self, 
                 input_dim: int = 4096,
                 output_dim: int = 768,
                 distance_type: Literal['cosine', 'l2'] = 'cosine',
                 weight_type: Literal['uniform', 'norm'] = 'uniform',
                 sinkhorn_epsilon: float = 0.1,
                 sinkhorn_max_iter: int = 100,
                 sinkhorn_threshold: float = 1e-7,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.vec_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        ).to(device=self.device, dtype=self.dtype)
        
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_threshold = sinkhorn_threshold
        self.distance_type = distance_type
        self.dist_func = (compute_distance_matrix_cosine 
                         if distance_type == 'cosine' 
                         else compute_distance_matrix_l2)
        self.weight_type = weight_type
        self.weight_func = (compute_weights_norm 
                           if weight_type == 'norm' 
                           else compute_weights_uniform)
    
    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super().to(device=self.device, dtype=self.dtype)
        # self.vec_transform = self.vec_transform.to(device=self.device, dtype=self.dtype)
        return self
        
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute OT loss between teacher and student outputs
        
        Args:
            student_outputs: Tensor of shape (batch_size, student_seq_len, student_dim)
            teacher_outputs: Tensor of shape (batch_size, teacher_seq_len, teacher_dim)
            
        Returns:
            loss: Scalar tensor representing the OT loss
        """
        batch_size = teacher_outputs.size(0)
        
        # Ensure inputs are on the correct device and dtype
        teacher_outputs = teacher_outputs.to(device=self.device, dtype=self.dtype)
        student_outputs = student_outputs.to(device=self.device, dtype=self.dtype)
        
        # Transform teacher outputs to match student dimension
        # teacher_outputs = self.vec_transform(teacher_outputs)
        
        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        for b in range(batch_size):
            teacher_seq = teacher_outputs[b]  # shape: (teacher_seq_len, dim)
            student_seq = student_outputs[b]  # shape: (student_seq_len, dim)
            
            # Compute cost matrix
            C = self.dist_func(teacher_seq, student_seq)
            
            # Compute weights
            weights1, weights2 = self.weight_func(teacher_seq, student_seq)
            
            try:
                # Convert to numpy arrays and ensure they are properly shaped
                C_np = to_numpy_safe(C)
                weights1_np = to_numpy_safe(weights1)
                weights2_np = to_numpy_safe(weights2)
                
                # Ensure arrays are contiguous and the correct shape
                C_np = np.ascontiguousarray(C_np)
                weights1_np = np.ascontiguousarray(weights1_np.reshape(-1))
                weights2_np = np.ascontiguousarray(weights2_np.reshape(-1))
                
                # Ensure arrays are float64 for numerical stability
                C_np = C_np.astype(np.float64)
                weights1_np = weights1_np.astype(np.float64)
                weights2_np = weights2_np.astype(np.float64)
                
                # Compute OT matrix
                P = ot.sinkhorn(
                    weights1_np, 
                    weights2_np, 
                    C_np,
                    reg=self.sinkhorn_epsilon,
                    numItermax=self.sinkhorn_max_iter,
                    stopThr=self.sinkhorn_threshold,
                    method='sinkhorn_stabilized'  # Use stabilized version
                )
                
                # Convert back to tensor
                P = torch.from_numpy(P).to(device=self.device, dtype=self.dtype)
                
                # Compute loss for current batch
                batch_loss = torch.sum(P * C)
                total_loss += batch_loss
                
            except Exception as e:
                print(f"Error in batch {b}: {str(e)}")
                print(f"Shapes - C: {C_np.shape}, weights1: {weights1_np.shape}, weights2: {weights2_np.shape}")
                raise e
            
        return total_loss / batch_size

def test_ot_loss():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16  # Changed from bfloat16 for better stability
    print(f"Using device: {device}, dtype: {dtype}")
    
    batch_size = 2
    teacher_seq_len = 10
    student_seq_len = 8
    teacher_dim = 768
    student_dim = 768
    
    teacher_outputs = torch.randn(batch_size, teacher_dim, 
                                device=device, dtype=dtype)
    student_outputs = torch.randn(batch_size,student_dim, 
                                device=device, dtype=dtype)
    
    print(f"Teacher outputs shape: {teacher_outputs.shape}")
    print(f"Student outputs shape: {student_outputs.shape}")
    
    ot_loss = OTLoss(
        input_dim=teacher_dim,
        output_dim=student_dim,
        distance_type='cosine',
        weight_type='norm',
        sinkhorn_epsilon=0.1,
        dtype=dtype,
        device=device
    )
    
    try:
        import time
        t1 = time.time()
        loss = ot_loss(student_outputs, teacher_outputs)
        t2 = time.time()
        print(f"Time taken: {t2 - t1:.4f} seconds")
        print(f"OT Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_ot_loss()