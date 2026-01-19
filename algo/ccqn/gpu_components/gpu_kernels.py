import torch

@torch.jit.script
def jit_solve_pgd(s: torch.Tensor, g: torch.Tensor, B: torch.Tensor, e_vec: torch.Tensor,
                  delta: float, cos_phi: float, max_iter: int) -> torch.Tensor:
    """
    Uphill PGD 求解器的 JIT 编译版。
    完全在 GPU 上运行，没有 Python 循环开销。
    """
    lr = 0.01
    
    # 预先计算投影所需的标量常量 (在 GPU 上)
    target_proj = delta * cos_phi
    # sqrt(1 - cos^2)
    sin_phi = torch.sqrt(torch.tensor(1.0, device=s.device, dtype=s.dtype) - cos_phi**2)
    target_perp = delta * sin_phi
    
    # 这个循环现在是在 C++ 层面执行的
    for _ in range(max_iter):
        # 1. 梯度下降 (Gradient Descent)
        # s = s - lr * (g + B @ s)
        grad = g + B @ s
        s = s - lr * grad
        
        # 2. 球面投影 (Trust Region Projection)
        norm_s = torch.norm(s)
        if norm_s > 1e-9:
            s = s * (delta / norm_s)
        else:
            s = e_vec * delta
            
        # 3. 圆锥投影 (Cone Projection)
        proj_len = torch.dot(s, e_vec)
        
        # 如果投影长度小于目标长度 (说明跑出了圆锥)
        if proj_len < target_proj:
            s_par = proj_len * e_vec
            s_perp = s - s_par
            norm_perp = torch.norm(s_perp)
            
            if norm_perp > 1e-9:
                # 重新组合：固定平行分量 + 拉伸垂直分量以保持总长 delta
                s = (target_proj * e_vec) + (target_perp * (s_perp / norm_perp))
            else:
                # 垂直分量丢失的极端情况，重置为轴向量
                s = e_vec * delta
                
    return s
