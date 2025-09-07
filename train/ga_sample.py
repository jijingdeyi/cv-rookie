# ---- 1. 准备：变量与工具函数 ----
import numpy as np
import torch
import torch.nn as nn
import pygad

# 7个权重，对应你的实际子损失
W = {
    "w_vf": 2.0,  # 可见光 MSE/SSIM 权重
    "w_if": 2.0,  # 红外 MSE/SSIM 权重
    "w_de": 2.0,  # 分解一致性权重
    "w_tv": 2.0,  # 融合TV/平滑约束权重
    "w_box": 1.0, # 检测 box 权重
    "w_cls": 0.5, # 检测 cls 权重
    "w_dfl": 0.5, # 检测 dfl 权重
}
KEYS = list(W.keys())

def _clamp_and_normalize(sol, low=1.0, high=10.0, total=10.0):
    sol = np.clip(np.array(sol, dtype=float), low, high)
    s = sol.sum()
    if s == 0:  # 兜底
        return np.ones_like(sol) * (total / len(sol))
    return sol / s * total

# 这几个均值会在每个 epoch 结束时被更新
mean_losses = {
    "w_vf": 0.5, "w_if": 0.5, "w_de": 0.5, "w_tv": 0.5,
    "w_box": 0.5, "w_cls": 0.5, "w_dfl": 0.5,
}

# ---- 2. 定义适应度函数（面向 pygad）----
def fitness_func(ga, solution, sol_idx):
    sol = _clamp_and_normalize(solution, 1.0, 10.0, 10.0)
    # 加权和：当前 epoch 的子损失均值 * 候选权重
    weighted = 0.0
    for wv, k in zip(sol, KEYS):
        weighted += wv * mean_losses[k]
    fitness = 1.0 / (abs(float(weighted)) + 1e-6)  # 越小越好 → 取倒数
    return fitness

# ---- 3. 构建 GA 对象（放在训练开始处）----
ga = pygad.GA(
    num_generations=5,
    num_parents_mating=5,
    sol_per_pop=5,
    num_genes=len(KEYS),
    gene_type=float,
    init_range_low=1.0, init_range_high=5.0,    # 初始范围
    parent_selection_type="rws",                 # 轮盘赌
    crossover_type="single_point",               # 单点交叉
    crossover_probability=0.25,
    mutation_type="random",
    mutation_by_replacement=True,
    random_mutation_min_val=1.0,
    random_mutation_max_val=5.0,
    mutation_percent_genes=10,                   # 10% 基因变异
    fitness_func=fitness_func,
    save_solutions=True,
    suppress_warnings=True
)

# ---- 4. 训练循环骨架 ----
for epoch in range(num_epochs):
    # (A) 清空收集器
    buf = {k: [] for k in mean_losses.keys()}  # 每类子损失的采样

    for batch in train_loader:
        # 前向：得到各子损失 (示例名，替换成你的实际变量)
        mse_V, mse_I = loss_mse_V, loss_mse_I
        decomp, tv   = loss_decomp, loss_tv
        box, cls, dfl = loss_box, loss_cls, loss_dfl

        # (B) 用上一轮权重 W 组装总损失
        total_loss = (
            W["w_vf"]*mse_V + W["w_if"]*mse_I +
            W["w_de"]*decomp + W["w_tv"]*tv +
            W["w_box"]*box + W["w_cls"]*cls + W["w_dfl"]*dfl
        )

        # 反传优化 ...
        # optimizer.zero_grad(); total_loss.backward(); optimizer.step()

        # (C) 收集用于 GA 的原子损失
        buf["w_vf"].append(mse_V.detach())
        buf["w_if"].append(mse_I.detach())
        buf["w_de"].append(decomp.detach())
        buf["w_tv"].append(tv.detach())
        buf["w_box"].append(box.detach())
        buf["w_cls"].append(cls.detach())
        buf["w_dfl"].append(dfl.detach())

    # (D) epoch 末：更新均值（建议压缩到0~1，防极端值）
    for k in buf:
        m = torch.stack(buf[k]).mean()
        mean_losses[k] = torch.sigmoid(m).item()  # or float(m)

    # (E) 运行 GA 搜索最佳权重
    ga.run()
    best_sol, best_fit, _ = ga.best_solution()
    best_sol = _clamp_and_normalize(best_sol, 1.0, 10.0, 10.0)

    # (F) 应用到下一 epoch 的权重
    for v, k in zip(best_sol, KEYS):
        W[k] = float(v)

    # (G) 可选：学习率调度/保存/早停等
    # scheduler.step(); save_ckpt(...)

print("Final weights:", W)
