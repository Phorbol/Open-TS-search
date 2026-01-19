# 重构 GPU 版 CCQN 优化器计划

## 1. 深度分析：现有代码存在的问题

通过对比您提供的 GPU 代码与项目现有的架构，我发现了以下几个关键的不合理之处：

1.  **架构风格不一致 (Monolithic vs Modular)**
    *   **现有架构**：采用了高度组件化的设计（`CCQNOptimizer` + `Components`），遵循单一职责原则。
    *   **用户代码**：是一个单体类 (`CCQNOptimizerGPU`)，将 Hessian 更新、模式选择、求解器逻辑全部耦合在主类中，导致难以维护和测试。
2.  **数据流管理混乱**
    *   代码中混合了 ASE (CPU/NumPy) 和 PyTorch (GPU/Tensor) 的操作。虽然必要的传输是不可避免的，但应当将其封装在边界处，而不是散落在算法逻辑中。
3.  **JIT 编译函数管理**
    *   `jit_solve_pgd` 作为全局函数定义，不仅污染命名空间，而且其中的参数（如 `max_iter=200`）被硬编码，缺乏灵活性。
4.  **CPU-GPU 同步瓶颈**
    *   在 PRFO 求解步骤中，使用 `scipy.optimize.brentq` (CPU) 来求解信赖域半径，这需要频繁地在 GPU 和 CPU 之间传输残差值，可能成为性能瓶颈。

## 2. 架构规划：组件化与融合

为了将代码融入现有架构，我建议**不直接修改现有的 CPU 版 `CCQNOptimizer`**，而是创建一个平行的 `CCQNGPUOptimizer`，并为其通过依赖注入的方式提供 GPU 版本的组件。

### 核心组件设计 (`algo/ccqn/gpu_components/`)

我们将把单体逻辑拆解为以下独立组件：

1.  **`GPUHessianManager`**:
    *   负责在 GPU 上维护 Hessian 矩阵 (`torch.Tensor`)。
    *   实现 GPU 版的 TS-BFGS 更新算法。
2.  **`GPUUphillSolver`**:
    *   封装 JIT 编译的 PGD 求解器。
    *   将 `jit_solve_pgd` 作为内部静态方法或独立内核管理。
3.  **`GPUPRFOSolver`**:
    *   实现 GPU 版的 PRFO 步长计算。
    *   尽量在 GPU 上完成信赖域子问题的求解（如果可能，实现一个简单的 GPU 二分查找替代 `brentq`，或者优化同步策略）。
4.  **`CCQNGPUOptimizer` (`algo/ccqn/ccqn_optimizer_gpu.py`)**:
    *   继承自 `ase.optimize.Optimizer`。
    *   **职责**：作为协调者，负责 ASE (CPU) 与 GPU 组件之间的数据桥接。
    *   **流程**：`Step Start` -> `CPU to GPU` -> `GPU Components Compute` -> `GPU to CPU` -> `Update Atoms`.

### 目录结构变动

```text
algo/ccqn/
├── ccqn_optimizer.py       # 现有 CPU 优化器
├── ccqn_optimizer_gpu.py   # [新建] GPU 优化器
├── components/             # 现有 CPU 组件
└── gpu_components/         # [新建] GPU 组件包
    ├── __init__.py
    ├── gpu_kernels.py      # [新建] 存放 JIT 编译的核心函数
    ├── hessian_manager.py  # [新建] GPU Hessian 管理
    ├── uphill_solver.py    # [新建] GPU Uphill 求解器
    └── prfo_solver.py      # [新建] GPU PRFO 求解器
```

## 3. 实施步骤

1.  **创建 GPU 内核模块**：将 `jit_solve_pgd` 等 JIT 函数提取到 `gpu_kernels.py`。
2.  **实现 GPU 组件**：依次实现 `GPUHessianManager`, `GPUUphillSolver`, `GPUPRFOSolver`，确保它们接受 Tensor 输入并返回 Tensor。
3.  **实现优化器主类**：编写 `CCQNGPUOptimizer`，组装上述组件。
4.  **注册与集成**：在 `registry/factory.py` 中注册新的优化器（如 `ccqn-gpu`），使其可以通过配置调用。

请确认是否同意采用这种**“组件化拆分 + 独立 GPU 优化器类”**的方案？如果同意，我将开始实施。
