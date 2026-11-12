# 神经网络均值—方差组合优化

MSML610 Fall2026 · 作者：bcai1-stack · [Issue #760](https://github.com/gpsaggese/gpsaggese.github.io/issues/760)

这个项目研究两种不同的学习方法：直接用神经网络近似受约束优化器，以及通过可微优化层训练有利于组合决策的收益预测器。评价包括求解精度、计算速度、样本外净收益与简单基线。

## 从哪里开始

1. 先读 [API 说明](mvo.API.md)，运行 [mvo.API.ipynb](mvo.API.ipynb)：解释目标、约束、梯度和训练接口。
2. 再读 [实验说明](mvo.example.md)，运行 [mvo.example.ipynb](mvo.example.ipynb)：完整实验、三种子回测、图表与不确定性分析。
3. 阅读 [实际运行报告](docs/experiment_report.md) 和 [测试记录](docs/validation.json)。

## 构建和启动

从本项目目录执行；宿主机按课程要求激活 thin environment，并启动 Docker：

```bash
DOCKER_ENGINE=docker ./docker_build.sh
DOCKER_ENGINE=docker ./docker_jupyter.sh
```

在浏览器打开终端中的 Jupyter 地址，进入 `/git_root/class_project/msml610/Fall2026/projects/UmdTask760_MSML610_Fall2026_Mean_Variance_Optimization_with_NN_1`。依次执行两个 Notebook 的全部单元。

`Dockerfile` 和 `requirements.txt` 与课程模板逐字相同。四个额外包由每份 Notebook 开头的固定版本安装单元提供。新建容器后再次运行该单元；Notebook 内核重启不会移除已安装包。

项目镜像名为 `bcai1-stack/msml610_mvo_nn`。构建脚本沿用模板的镜像和环境配置，将构建日志留在项目目录内，并支持无交互终端。其他 Docker 辅助脚本来自模板。默认模板在 macOS 上选择 Apple container，因此这里显式设置 `DOCKER_ENGINE=docker`。

## 验证

```bash
# 自动启动一个独立临时 Docker 容器，安装同样的四个包并运行测试。
./run_tests.sh

# 加上两个 Notebook 的全单元执行；每份使用全新内核。
./run_tests.sh --notebooks
```

也可以在已安装依赖的容器中运行 `python validate.py --notebooks`。测试覆盖解析解、可微层梯度的数值差分、换手限制、历史特征隔离、训练标准化、费用和权重漂移、数据解析，以及近似网络和滚动回测的完整流程。

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
| `mvo/optimization.py` | 同一目标和约束的普通求解器与可微层 |
| `mvo/amortized.py` | 权重近似网络、换手可行性、误差与公平计时 |
| `mvo/data.py` | 数据下载、输入校验、历史特征和协方差 |
| `mvo/learning.py` | 两种训练损失、标准化、验证选模 |
| `mvo/backtest.py` | 时间窗口、六策略比较、输出与图表 |
| `mvo/metrics.py` | 漂移与费用、绩效、配对区块 bootstrap |
| `mvo/reporting.py` | 从实际结果自动生成中文报告 |
| `tests/`、`validate.py` | 单元/集成测试与 Notebook 执行验证 |
| `docs/experiment_snapshot.json` | 配置、来源哈希、逐折时间与结果 |
| `docs/video_outline.md` | 10–20 分钟个人演示提纲 |

原来的模板和手动探索单元保存在本机 `drafts/`，不加入提交。`data/` 和 `results/` 同样被忽略；执行 Notebook 可以重新生成。

## 命令行入口

在项目目录对应的容器工作目录运行：

```bash
python -m mvo amortized
python -m mvo amortized --turnover-limit 0.4 --output results/turnover_cli
python -m mvo download
python -m mvo backtest --csv data/french10.csv --output results/french_cli
```

省略 `--csv` 的回测使用模拟数据，仅用于检查流程。

## 解释边界

行业组合不是十只可直接买卖的 ETF；数据库可能修订历史收益。本项目没有独立的最终保留市场时期，也没有整数手数或持仓数量约束。三个种子共享市场历史，不能当作独立市场样本。计算加速要连同解误差和训练成本一起解读；学习方法未必优于简单基线。

[相关研究](docs/related_projects.md) · [原始项目规格](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Mean_Variance_Optimization_with_NN.md)
