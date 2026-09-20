# PI05 basket：初始 loss、BF16 梯度与 relative action

本文保留 9 月 18 日的调查快照。FP32 完整运行和选择性 FP32 的后续分组验证
见 [9 月 20 日报告](pi05_selective_precision_20260920.md)。

当前修改在 `codex/pi05-basket-instability-20260918`，基于 `6dbfc8edb0ed`。
另外取得并检查了 `786f06f6907c866a458c796749245a32bc182f13`。
runner、指标和训练入口未修改，也没有新增 target batch 参数。
保留工作区已有的 `/mnt/data/oss-wlcb/raw_data/...` 数据路径修改。

## 两个现象需要分开判断

| 运行          | 第一步 raw loss | 后续观测                                                     |
| ------------- | --------------- | ------------------------------------------------------------ |
| `6dbfc8` FP32 | 1.262672        | 第 289 步 raw loss 0.333252，平滑 loss 0.510604              |
| `786f06` BF16 | 1.280878        | 14,001–16,000 步均值 0.019269；19,001–19,994 步均值 0.159061 |

这是读取日志时的快照；FP32 运行仍在继续。两份日志均未出现 NaN/Inf。
BF16 完成的第一轮为 11,763 步，后期退化不与 epoch 切换重合。
这里的 loss 是原日志的 rank 0 指标，不是跨 rank 平均。

初始 1.x 本身不能判定崩溃。Flow matching 的目标是 `noise - actions`，
噪声方差约为 1；未经学习的动作头出现这个数量级并不异常。
两次运行都用原生 42 维动作，却加载 32 维 base checkpoint：
`action_in_proj.weight`、`action_out_proj.weight/bias` 因形状不同重新初始化，
匹配的其他权重正常加载。两次运行的 warmup 都是 1,000 步。

![loss 对照](../work_dirs/pi05_loss_diagnosis_20260918/loss_comparison.png)

## 已定位并修复的精度问题

`openpi_fp32_flow` 保护了 timestep MLP，但 AdaRMS 的条件投影
`dense(cond)` 仍在外层 autocast 内执行。这会把 scale、shift、gate 的计算
降到 BF16，改变回传到 timestep MLP 的梯度。

固定 `786f06` 第 11,763 步 checkpoint、两个真实 basket 样本、图像、
token、action、noise 和 `t=0.1`，以同一模型的纯 FP32 计算为参照：

| 路径          | loss     | timestep MLP 权重梯度相对 L2 误差 | 梯度方向余弦 |
| ------------- | -------- | --------------------------------- | ------------ |
| FP32 参照     | 0.076750 | 0                                 | 1            |
| `786f06` BF16 | 0.076849 | 1.23065                           | 0.50140      |
| 修复后的 BF16 | 0.076709 | 0.03206                           | 0.99950      |

只把 SDPA attention 改成 FP32 没有消除这一差异。
保护 AdaRMS 条件投影后，`t=0.01/0.5/0.9` 的梯度误差也下降。
以上前后比较保持 action 表示和 normalization 完全一致，因此这个数值问题
可以独立于 relative action 验证。该定点比较关闭 RTC 以固定时间条件；
逐位置 RTC 条件另由回归测试和完整 runner 的更新步骤覆盖。

修改包括：

- Gemma 增加 `adarms_fp32`，basket action expert 开启；其他配置默认关闭。
  条件投影显式使用 FP32，参数名称和 checkpoint 结构不变。
- 保留 `786f06` 的 FP32 residual stream 修正：FP32 master 参数下，
  Gemma/SigLIP 输入不再被强制转换为 BF16，矩阵计算仍由 autocast 控制。
- 保留 `6dbfc8` 的 FP32 attention mask 修复，因此纯 FP32 对照仍能运行。

这些证据确认并修正了数值偏差，**尚不能证明它是后期退化的唯一原因，
也没有完成修复后跨越原退化区间的完整复训**。
旧 64 维运行第 70,578 步 checkpoint 在固定真实样本 `t=0.1` 上，
改用纯 FP32 后 loss 仍为 0.88472：已经退化的权重不会仅靠换精度恢复。

## relative action 的处理

数据 metadata 定义的 state 是 33 维，action 是 42 维。

| action 索引（从 0 开始） | 内容              | 转换                      |
| ------------------------ | ----------------- | ------------------------- |
| 0–30                     | 31 个关节位置指令 | 减去当前观测 `state[:31]` |
| 31–39                    | 底座位置和 rot6d  | 保持数据原表示            |
| 40–41                    | 左右手开关        | 保持绝对开关值            |

`state[31:33]` 是手部开关，并不是 `action[31:33]` 的底座坐标。
因此不能对前 33 维直接相减，也不能把 42 维全部当成同名 state。
所有未来动作都减同一个当前 state；不是相邻动作作差。

新增 `configs/pi05/pi05_paligemma_basket_all_rtc_relative_fp32_full_finetune.py`：
在 normalization 前插入已有的 `RelativeActions(mask=[True] * 31)`，
并使用同一 mask 重新计算 action-window statistics。
统计工具原来无条件拒绝 state/action 维度不等的 relative 转换，现改为
检查 mask 是否超出两者可用维度，支持这个合法的共享前缀。

已实际计算全部 14 个数据目录、2,899 个 episode、1,505,539 帧、
48,177,248 个 action-window 目标的统计量。
前 31 维未归一化标准差相对原表示的比例中位数为 0.57444，范围
0.39174–0.82865；state 和 action 尾部统计量在浮点容差内保持一致。
六个真实样本的相对动作反变换最大误差为 `1.79e-7`，
图像、token、mask、noise 完全相同。

这表明 relative action 是合理的建模对照，不能据此证明它能单独防止崩溃。
公开 LeRobot 参照版本的 `use_relative_actions` 默认也是 False；
用户所述稳定运行的实际配置尚未取得。

部署 relative 策略时，必须先反归一化，再给动作前 31 维加回当前 state。
relative 实验应从 base 初始化，并使用新 work directory 和新统计量。

## 验证与复跑

52 项定向 pytest 通过：AdaRMS FP32/BF16 和 scalar/RTC 梯度、
33/42 维统计与正反变换、已有 PI0/PI05 训练推理、LeRobot 核心对齐、
时间采样和 FSDP checkpoint。通过 flake8、isort 和 diff 检查。

完整模型验证使用两张 A800、真实数据、RTC、FP32 master、AdamW、EMA，
通过原 FSDP runner 执行。每 rank microbatch 2，各连续执行 3 次更新。
BF16 修复、绝对动作 FP32、relative action FP32 三条路径均完成更新，
loss、裁剪前梯度范数均有限，每个 rank 的 Adam step 均为 3。
这验证执行路径，并非长程稳定性结论。

现有纯 FP32 任务已经在下降，先观察它经过 warmup 后的曲线。
要验证 BF16 数值修复，使用当前工作区的主 basket 配置从 base 做对照。
要单独验证 relative action，使用以下配置：

```bash
bash scripts/train.sh \
  configs/pi05/pi05_paligemma_basket_all_rtc_relative_fp32_full_finetune.py \
  ./work_dirs/pi05_basket_relative_fp32_control \
  --cfg-options train_dataloader.per_device_batch_size=8 \
  runner.max_epochs=6 runner.max_steps=None
```

沿用训练平台的分布式环境配置。当前梯度累积为 1，有效 batch 是
`8 × 总 rank 数`：16 ranks 为 128，32 ranks 为 256。
设置 `max_epochs=6` 不会自动改变配置中 `decay_steps=94104` 的 LR 曲线。

实验脚本、固定真实 batch、精度对照 JSON、全量 relative statistics、
训练日志与测试输出均位于 `work_dirs/pi05_loss_diagnosis_20260918/`。
`real_batch_probe.py` 记录了旧路径的模拟方式，`training_step_probe.py`
只在独立脚本中采集诊断，不修改生产 runner。
