# CCQN E-Vector Generator Implementation Plan (Compatible)

本計劃將實現基於「無記憶正則化投影」的 `EVectorGenerator`，並確保與現有代碼完全兼容。

## 1. 新建模塊 (`EVectorGenerator`)

文件：`algo/ccqn/gpu_components/e_vector_generator.py`

### 功能

* 實現 `compute_cone_axis_robust` 算法。

* 解析帶有 `intent` 的 `reactive_bonds`。

## 2. 修改 `CCQNGPUOptimizer`

文件：`algo/ccqn/ccqn_optimizer_gpu.py`

### 2.1 初始化變更

* 新增參數 `use_robust_e_vector` (bool, default=False) 或檢測 `reactive_bonds` 格式。

* 若啟用，實例化 `EVectorGenerator`。

### 2.2 執行邏輯變更 (`step`)

```python
if self.e_vector_generator:
    # 新模式：Robust Markovian Projection
    e_vec_np = self.e_vector_generator.compute(self.atoms, f)
else:
    # 舊模式：Legacy Democratic/Local (保持不變)
    e_vec_np = self._calculate_e_vector_cpu(f, x_k_np)
```

## 3. 兼容性保證

* 默認情況下（不傳新參數，使用舊格式 `reactive_bonds`），代碼路徑完全不變，執行 `_calculate_e_vector_cpu`。

* 只有顯式啟用新功能時，才會切換到新邏輯。

## 4. 驗證

* 運行現有的 H2 測試腳本，確保在默認設置下結果與之前完全一致（回歸測試）。

