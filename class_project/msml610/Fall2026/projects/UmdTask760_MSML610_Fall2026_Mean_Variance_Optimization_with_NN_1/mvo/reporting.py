"""Write a readable report from executed experiment results."""

from pathlib import Path
import shutil


def amortized_section(results):
    lines = ["## 实验 A：近似误差和计算时间", "",
             "| 设置 | 平均目标差距 | 权重 RMSE | 网络单次 ms | 求解器单次 ms | 收支平衡次数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in results.items():
        count = metrics["break_even_instances"]
        count = str(count) if count is not None else "无（单次未加速）"
        lines.append(f"| {name} | {metrics['objective_gap_mean']:.5f} | "
                     f"{metrics['weight_rmse']:.5f} | {metrics['nn_single_median_ms']:.4f} | "
                     f"{metrics['solver_single_median_ms']:.4f} | {count} |")
    lines += ["", "时间受硬件和系统负载影响。收支平衡次数只比较计算成本，忽略近似误差；"
              "网络批量吞吐量不能直接当作单次延迟。不同约束的目标差距也不能直接互相排名。", ""]
    return lines


def backtest_section(reports):
    strategies = next(iter(reports.values()))["metrics"]
    lines = ["## 实验 B：三个预先指定种子的均值", "",
             "| 策略 | CAGR | 波动率 | Sharpe | 最大回撤 | 净效用 | 年化换手 | 最大目标权重 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    averages = {}
    for strategy in strategies:
        values = [report["metrics"][strategy] for report in reports.values()]
        mean = {key: sum(value[key] for value in values) / len(values) for key in values[0]}
        averages[strategy] = mean
        lines.append(f"| {strategy} | {mean['cagr']:.2%} | {mean['annual_volatility']:.2%} | "
                     f"{mean['sharpe_rf_zero']:.3f} | {mean['max_drawdown']:.2%} | "
                     f"{mean['annual_utility']:.4f} | {mean['annual_turnover']:.2f} | "
                     f"{mean['maximum_target_weight']:.2%} |")
    best = max(averages, key=lambda name: averages[name]["annual_utility"])
    lines += ["", f"本次平均净效用最高的策略是 `{best}`。", ""]
    for baseline in ("equal_weight", "minimum_variance", "predict_then_optimize", "shrunk_mean_mvo"):
        wins = sum(report["metrics"]["decision_focused"]["annual_utility"] >
                   report["metrics"][baseline]["annual_utility"] for report in reports.values())
        lines.append(f"- 决策导向净效用高于 `{baseline}`：{wins}/{len(reports)} 个种子。")
    return lines + ["", "种子共享市场历史，不能作为独立市场样本。收缩对照不能严格分解神经网络提升的因果来源。", ""]


def uncertainty_section(report):
    intervals = report["decision_focused_minus_baseline_utility_ci"]
    lines = ["## 不确定性（固定 seed=7）", "",
             "| 决策导向减去基线 | 95% 区块 bootstrap 下界 | 上界 |",
             "|---|---:|---:|"]
    for name, interval in intervals.items():
        lines.append(f"| {name} | {interval['lower_95']:.4f} | {interval['upper_95']:.4f} |")
    if all(interval["lower_95"] <= 0 <= interval["upper_95"] for interval in intervals.values()):
        lines += ["", "所有区间跨零，本次检查未显示稳健的决策导向优势。"]
    return lines + ["", "区块 bootstrap 不能消除市场非平稳性和模型选择偏差。", ""]


def write_report(snapshot, root):
    root = Path(root)
    report = snapshot["backtests"]["7"]
    data = snapshot["data"]
    lines = ["# 实际运行结果与解释", "",
             "数值由完整实验自动生成。配置、哈希与逐折结果见 `experiment_snapshot.json`；"
             "测试环境和执行记录见 `validation.json`。", "",
             f"输入：{data['start']} 至 {data['end']}，{data['rows']} 个交易日、10 个行业组合。",
             f"样本外：{report['test_start']} 至 {report['test_end']}，"
             f"{report['test_days']} 个交易日、{len(report['folds'])} 折。"
             "末尾不足完整持有期的日期不进入评价。", ""]
    lines += amortized_section(snapshot["amortized"])
    lines += backtest_section(snapshot["backtests"])
    lines += uncertainty_section(report)
    lines += ["![固定种子7的净值和回撤](backtest_seed7.png)", "", "## 局限", "",
              "- 训练采用毛效用，评价采用扣费后净收益；二者并不完全一致。",
              "- 没有整数手数/持仓数量限制，也没有另设完全未触碰的最终市场测试期。",
              "- 行业组合是研究序列，存在历史修订，不能等同于可交易 ETF 的实盘回报。",
              "- CVXPYlayers/NumPy 接口出现上游弃用警告；当前数值和梯度检查通过。", ""]
    (root / "docs/experiment_report.md").write_text("\n".join(lines))
    shutil.copyfile(root / "results/french_seed7/backtest.png", root / "docs/backtest_seed7.png")
