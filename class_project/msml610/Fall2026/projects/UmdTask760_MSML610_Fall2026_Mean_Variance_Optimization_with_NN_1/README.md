# 神经网络均值—方差组合优化

MSML610 Fall2026 · 作者：bcai1-stack · [Issue #760](https://github.com/gpsaggese/gpsaggese.github.io/issues/760)

这个项目研究两种不同的学习方法：直接用神经网络近似受约束优化器，以及通过可微优化层训练有利于组合决策的收益预测器。评价包括求解精度、计算速度、样本外净收益与简单基线。

## 从哪里开始

全部 Python 实现、演示和测试都保存在 `.ipynb` 文件中，可以在 Jupyter 内逐个单元格阅读和修改。

1. 先读 [API 说明](mvo.API.md)，运行 [mvo.API.ipynb](mvo.API.ipynb)：解释目标、约束、梯度和训练接口。
2. 打开 [mvo_utils.ipynb](mvo_utils.ipynb)：查看实际函数实现，按优化器、数据、训练、回测等主题逐段学习。
3. 再读 [实验说明](mvo.example.md)，运行 [mvo.example.ipynb](mvo.example.ipynb)：完整实验、三种子回测、图表与不确定性分析。
4. 运行 [mvo.tests.ipynb](mvo.tests.ipynb) 检查修改；阅读 [实际运行报告](docs/experiment_report.md) 和 [测试记录](docs/validation.json)。

API、example 和测试 Notebook 会通过 `%run ./mvo_utils.ipynb` 加载同一份函数。每份都能从新内核独立运行，不需要先在另一个 Notebook 中执行代码。

## 构建和启动

从本项目目录执行；宿主机按课程要求激活 thin environment，并启动 Docker：

```bash
DOCKER_ENGINE=docker ./docker_build.sh
DOCKER_ENGINE=docker ./docker_jupyter.sh
```

在浏览器打开终端中的 Jupyter 地址，进入 `/git_root/class_project/msml610/Fall2026/projects/UmdTask760_MSML610_Fall2026_Mean_Variance_Optimization_with_NN_1`。依次执行 API 和 example Notebook 的全部单元。

`Dockerfile` 和 `requirements.txt` 与课程模板逐字相同。四个额外包由 `mvo_utils.ipynb` 开头的固定版本安装单元提供，入口 Notebook 的 `%run` 会一并执行它。新建容器后再次运行；Notebook 内核重启不会移除已安装包。

项目镜像名为 `bcai1-stack/msml610_mvo_nn`。构建脚本沿用模板的镜像和环境配置，将构建日志留在项目目录内，并支持无交互终端。其他 Docker 辅助脚本来自模板。默认模板在 macOS 上选择 Apple container，因此这里显式设置 `DOCKER_ENGINE=docker`。

## 验证

```bash
# 自动启动一个独立临时 Docker 容器，通过 Notebook 安装依赖并运行测试。
./run_tests.sh

# 加上 API 和 example 的全单元执行；每份使用全新内核。
./run_tests.sh --notebooks
```

也可以直接在 Jupyter 中打开 `mvo.tests.ipynb`，从上到下运行全部单元，查看 15 项单元与集成测试。`mvo.validation.ipynb` 负责使用新内核执行 Notebook、保存执行输出，并记录环境与代码哈希；`run_tests.sh` 是它的 Docker 启动入口。

测试覆盖解析解、可微层梯度的数值差分、换手限制、历史特征隔离、训练标准化、费用和权重漂移、数据解析，以及近似网络和滚动回测的完整流程。当前验证状态、执行时间和检查范围以 `docs/validation.json` 为准。

Notebook 和测试使用 CPU，不需要 GPU。历史数据下载需要联网；下载失败会明确报错。已有数据缓存须通过 SHA-256 校验。安装依赖和下载所需时间取决于网络。

## 研究设置

- 目标：最小化 `γ × 年化方差 − 年化预期收益`，固定 `γ=5`。
- 基础约束：只做多、满仓。近似实验另比较 L1 换手上限 0.4。
- 实验 A：512 个训练实例、128 个独立测试实例、64×64 MLP、150 轮；输入包括均值、协方差和上期权重。
- 实验 B：French 十行业价值加权日收益，2010—2025；126 日历史窗口，21 日持有期；60/12/12 期训练/验证/测试窗口向前滚动。
- 六个策略：等权、最小方差、历史 MVO、均值收缩 MVO、预测后优化、决策导向训练。
- 学习方法共享 32 单元 Tanh 网络，均值输出限制在 ±0.25，最多 15 轮，验证选轮次；使用种子 7、11、19。
- 回测按每买入或卖出一美元收费 10 bps；跨折保持漂移后的持仓，初始现金配置也收费。

训练效用以毛收益和持有期方差计算；净收益评价扣除交易成本。两者并不完全相同。换手上限用于独立的近似实验，六策略历史回测保持相同的非负满仓可行域。

## 文件索引

| 文件 | 职责 |
|---|---|
| `mvo_utils.ipynb` | 所有实现：优化器、数据、训练、近似实验、回测、指标与报告生成 |
| `mvo.API.ipynb` | 小例子、解析解、梯度与换手约束演示 |
| `mvo.example.ipynb` | 数据、完整实验、三个种子、图表和结果解释 |
| `mvo.tests.ipynb` | 15 项单元与集成测试，保存测试日志 |
| `mvo.validation.ipynb` | 在新内核执行 Notebook，记录环境、代码哈希和验证结果 |
| `run_tests.sh` | 使用临时 Docker 容器运行验证 Notebook |
| `docs/experiment_snapshot.json` | 配置、来源哈希、逐折时间与结果 |
| `docs/video_outline.md` | 10–20 分钟个人演示提纲 |

Docker 的 `.sh` 脚本、`Dockerfile`、`requirements.txt` 和 Markdown 说明继续保留各自格式；项目的 Python 逻辑由 Notebook 管理。

原来的模板和手动探索单元保存在本机 `drafts/`，不加入提交。`data/` 和 `results/` 同样被忽略；执行 Notebook 可以重新生成。

## 如何修改代码

1. 在 `mvo_utils.ipynb` 中找到对应主题，阅读函数上方的说明，再修改代码单元并保存。
2. 回到要运行的 API、example 或测试 Notebook，选择 **Kernel → Restart Kernel and Run All Cells**。这样会重新加载保存后的函数，也能检查是否依赖了旧内核中的变量。
3. 先运行 API 与测试，确认小例子和约束检查通过，再运行完整 example；完整回测需要更长时间。
4. 用 `./run_tests.sh --notebooks` 完成一次独立容器验证，检查更新后的测试记录和实验报告。

只在另一个 Notebook 中修改变量，不会自动更新当前内核；修改函数后必须保存并重新加载。研究配置（例如随机种子、训练轮数和回测窗口）可在 `mvo.example.ipynb` 的实验单元中调整。`mvo_utils.ipynb` 定义函数，不会在加载时自动运行完整回测。

## 解释边界

行业组合不是十只可直接买卖的 ETF；数据库可能修订历史收益。本项目没有独立的最终保留市场时期，也没有整数手数或持仓数量约束。三个种子共享市场历史，不能当作独立市场样本。计算加速要连同解误差和训练成本一起解读；学习方法未必优于简单基线。

[相关研究](docs/related_projects.md) · [原始项目规格](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Mean_Variance_Optimization_with_NN.md)
