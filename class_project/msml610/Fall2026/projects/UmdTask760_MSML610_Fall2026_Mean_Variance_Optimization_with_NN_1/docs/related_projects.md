# 相关研究与项目定位

本项目区分两类模型：网络直接输出近似最优权重，和网络经可微求解器输出权重。后一类每次仍需调用求解器，不能将其与前一类的摊销加速混为一谈。

| 参考项目 | 相关内容 | 本项目的边界 |
|---|---|---|
| [CVXPYlayers](https://github.com/cvxpy/cvxpylayers) | 把参数化凸问题作为可微层，支持 PyTorch 的前向求解和反向求导 | 使用它实现决策导向训练；另行测量直接预测权重的网络速度 |
| [DeepDow](https://github.com/jankrepl/deepdow) | 用深度学习进行组合配置，提供可微配置层和组合损失 | 采用较小的固定网络，以透明的六策略对照为主 |
| [E2E-DRO](https://github.com/Iyengar-Lab/E2E-DRO) | 将预测与分布鲁棒优化结合，并研究从数据学习风险偏好和鲁棒程度 | 当前使用普通均值—方差问题和固定风险偏好，没有实现分布鲁棒优化 |

## 方法来源

- [Agrawal et al., Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)：可微凸优化层的原理。
- [Costa and Iyengar, Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134)：组合场景中的端到端学习和分布鲁棒扩展。
- [Kenneth French 十行业组合定义](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_ind_port.html)：本项目真实历史收益的来源和构造。

没有将其他项目的性能数字当成本项目结果。本地数值全部来自当前实现的运行记录。简单等权、最小方差和均值收缩对照用于检验复杂模型是否有额外价值。
