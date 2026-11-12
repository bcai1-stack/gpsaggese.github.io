# MVO API 说明

配套 Notebook：[mvo.API.ipynb](mvo.API.ipynb)。从新内核按顺序运行全部单元；开头的 `%run ./mvo_utils.ipynb` 会安装固定版本的额外依赖并加载函数。

所有接口实现都在 [mvo_utils.ipynb](mvo_utils.ipynb) 中，每个主题包含说明和可编辑的代码单元。修改后先保存，再重启 API Notebook 的内核并运行全部单元，确保使用新实现。

数值输入是年化收益均值 `mean`（n 维）和年化协方差 `covariance`（n×n）。

## 核心接口

| 接口 | 输入与返回 |
|---|---|
| `covariance_factor(covariance)` | 校验对称半正定矩阵，返回满足 `F.T @ F = covariance` 的因子 |
| `MeanVarianceSolver(n_assets, risk_aversion=5, turnover_limit=None)` | 编译可复用的 CVXPY 问题，使用 CLARABEL 求解 |
| `solver.solve(mean, covariance, previous=None)` | 返回非负且和为 1 的权重；有换手限制时必须传入满仓的 `previous` |
| `make_allocation_layer(n_assets, risk_aversion=5, turnover_limit=None)` | 返回 PyTorch 可微层，参数顺序为 `(mean, factor)`；有换手限制时追加 `previous` |
| `make_samples(frame, lookback=126, horizon=21)` | 生成严格历史特征、协方差因子、均值和未来持有期标签 |
| `train_model(train, validation, mode, ...)` | `mode='mse'` 或 `'utility'`；仅训练集拟合标准化，验证集选择 epoch |

目标是 `risk_aversion * ||F @ w||² - mean @ w`。非负与满仓始终成立；换手上限是 `sum(abs(w - previous)) <= turnover_limit`，合法上限在 `(0, 2]`。

因子通过特征分解构造，协方差非正定但半正定时也可处理。小数值负特征值按容差截断，明显非对称或有负特征值的输入会报错。模型没有静默修复不合法金融数据。

## 可微求解

普通求解器和可微层共享问题定义。可微层要求 double 精度的张量用于当前训练路径。API Notebook 对单个权重反向传播；对所有权重之和反向传播会因满仓约束接近零梯度。

[mvo.tests.ipynb](mvo.tests.ipynb) 的有限差分测试在内部可微的二资产例子中验证梯度，而不是仅检查 `backward()` 没有报错。约束切换处解映射可能不可微，因此不能把一个数值例子的验证推广到所有边界点。

## 数据约定

日收益用小数表示，例如 1% 写成 0.01；日期严格递增且不重复；至少两项资产，收益有限且大于 -1。协方差使用历史样本的 Ledoit–Wolf 收缩估计。训练和回测统一使用年化系数 252。

参考：[CVXPY](https://www.cvxpy.org/)、[CVXPYlayers](https://github.com/cvxpy/cvxpylayers)、[PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html)。
