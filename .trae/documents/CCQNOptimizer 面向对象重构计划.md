## 背景与目标
- 保持 CCQNOptimizer 的数值行为、接口与日志完全不变
- 将“模式切换、Hessian 管理、方向(e-vector)生成、步长求解、信任域更新、收敛判据”等职责解耦
- 统一数值容差、PBC/MIC 处理与形状管理，避免隐式假设与形状错误

## 架构设计
- OptimizerCore: 保留 ASE Optimizer 兼容入口，协调各子模块；对外接口不变（构造参数、step/converged、logfile、fmax 等）
- Config(不可变): 统一保存并校验 e_vector_method、cos_phi、trust 半径、ic_mode、idpp_images、use_idpp、hessian、阈值/容差
- StateTracker: 跟踪 pos_k_minus_1、g_k_minus_1、energy_k_minus_1、rho；提供能量预测与步长历史
- HessianInitializer: 从 calc 或缩放单位阵初始化；集中容错与维度检查
- HessianUpdater(TS-BFGS): 纯函数式更新；包含失败回退逻辑与对称性保持
- ModeSelector: 基于最小特征值判定 uphill/prfo；包含对角化失败回退与日志
- DirectionProvider
  - InterpStrategy: 线性或 IDPP 路径生成至中点，MIC 位移展平为 e-vector
  - ICStrategy: democratic/weighted 两模式；PBC/MIC；零范数与力投影回退
- StepSolver
  - UphillSolver: 约束二次规划（SLSQP）+ 回退（e_vec*r）；保持等式/不等式约束和容差
  - PRFOSolver: 特征基无约束步长、RFO 子问题、brentq 约束半径求解、最终范数保护与缩放回退
- TrustRegionManager: 线性半径 r 的更新（sqrt(sigma) 因子）；边界步判定；最小/最大半径钳制
- ConvergenceChecker: 统一 fmax 计算、形状自适应与“必须进入 prfo 模式”的判据
- Logger: 统一格式化与 flush；保证现有文案与时机一致

## 数据与数值约束
- 形状规范: forces(N,3)/flatten(3N,)、positions(N,3)/flatten(3N,)；显式 reshape
- 容差统一: eps_machine、tol、ftol、rcond、最小范数阈值；集中定义并复用
- PBC/MIC 一致性: find_mic 与向量化 IDPP 的最短镜像向量实现完全对齐
- 特征分解健壮性: eigh 失败回退到单位阵；特征符号处理与绝对值使用严格复现
- 约束求解健壮性: SLSQP 失败时严格回退；brentq 区间与容差与现有逻辑一致

## 对外 API 保持
- 构造参数与默认值完全一致；同名字段沿用
- e_vector_method='interp'/'ic'、ic_mode、use_idpp/idpp_images 等语义不变
- 日志内容、字段与触发点保持一致；文件写入与 flush 不变

## 迁移步骤
1. 引入 Config、StateTracker 的骨架（仅内部使用），将现有属性收口
2. 抽取 HessianInitializer/Updater，替换内联初始化与更新
3. 抽取 ModeSelector，替换 step 中的特征分解与模式切换
4. 抽取 DirectionProvider 两策略，替换 _calculate_e_vector，复用 robust_interpolate 与 Vectorized_ASE_IDPPSolver
5. 抽取 UphillSolver、PRFOSolver，分别替换 _solve_uphill_step/_solve_prfo_step/_solve_rfo_subproblem
6. 抽取 TrustRegionManager 与 ConvergenceChecker，替换对应内联逻辑
7. 整理 Logger 调用点，保证文案和时机与原代码一致

## 等价性验证
- 单步等价: 在固定原子、力与 Hessian 下，比较每个子模块输出与原实现逐元素相同（或在浮点容差内）
- 模式切换等价: 基于相同 Hessian 最小特征值序列，模式变更时机一致
- 信任域等价: r 的更新序列完全一致；边界步判定等价
- e-vector 等价: interp/IDPP 与 IC 两模式输出方向一致（容差内）
- 收敛判据等价: fmax 与形状自适应处理一致
- 日志等价: 行文、数值与触发时机一致

## 测试计划
- 单元测试: Hessian 初始化/更新、RFO 子问题、brentq 约束、SLSQP 回退、MIC 位移
- 参数化测试: ic_mode 两策略、use_idpp on/off、不同 trust 半径
- 回归测试: 真实或合成体系上跑固定步数，比较模式序列、步长范数、rho 与日志
- 边界测试: 零向量、奇异 Hessian、SLSQP 失败、eigh 失败、brentq 无解回退

## 风险与对策
- 数值顺序差异: 控制容差并在最终缩放处钳制，保持范数与边界一致
- PBC 差异: 统一 MIC 路径与向量化实现，避免隐含换算误差
- 依赖行为差异: 固定 SciPy/ASE 版本接口假设；必要时做版本特征检测

## 交付物
- 模块化类的实现骨架与文档说明（不改变外部接口）
- 等价性测试套件与日志比对工具
- 旧→新职责映射表、关键数值路径说明