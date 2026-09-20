# PI05 basket：选择性 FP32 与完成的 FP32 对照

后续反例：`9fa2cefebe` 的同 batch 256 长跑在约 11,000 步后仍然退化。
因此本文的 AdaRMS 修复不足以保证稳定；下文保留当时的短程测量记录。
新发现、attention 内核对照与进一步修正见
[后续诊断](pi05_attention_instability_20260920.md)。

本次在 `fix/lyh/fix-pi05-loss-explode`（HEAD `6dbfc8edb0ed`）的工作区验证。
旧 commit 本身不包含工作区的 AdaRMS 修复。兼容性审查后，修复放入独立的
`pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune.py`，明确启用
BF16 autocast 并保留少数模块的 FP32 计算；原主配置和全 FP32 配置保留旧默认。
没有修改 runner、batch 累积、优化器、学习率或动作表示。

## 完整运行的证据与限制

| 运行                                   | 更新数 | 完成 epoch | 最后 1,000 步 raw loss 均值 |
| -------------------------------------- | -----: | ---------: | --------------------------: |
| `786f06f690`，BF16，global batch 128   | 19,994 |       1.70 |                    0.159083 |
| `6dbfc8edb0ed`，FP32，global batch 256 | 35,292 |       6.00 |                    0.010310 |

两次的 `dataset_statistics.json` 内容完全相同，都是 42 维原生动作，
没有增加关节 relative action 转换；已保存的模型与训练配置除精度开关和
统计文件路径外一致。但两次每个 epoch 分别有 11,763 和 5,882 次更新，
有效 batch 不同，不能把完整运行当成严格的单变量精度实验。
两份日志均无 NaN/Inf，BF16 表现为 loss 退化，而非非有限数值。

![训练曲线](../work_dirs/pi05_selective_precision_20260920/loss_comparison.png)

## 用相同权重和输入定位模块

固定同一个模型的权重、两个真实 basket 样本、noise、时间和 RTC 随机种子，
每条路径独立重新执行 forward/backward，不更新权重。比较下列两个 checkpoint
的 `train_model`，而不是 EMA：

- BF16 第 11,763 步，尚未进入日志中的退化区间。
- 已完成的 FP32 第 35,292 步。

以同一 checkpoint 的 FP32 梯度为参照；预先设定时间 MLP 两个 weight 的
相对 L2 误差不超过 0.1、方向余弦至少 0.99。检查普通 `t=0.1`、RTC
`t=0.1` 和 RTC `t=0.5`。BF16 参照保留 `786f06` 的 FP32 residual stream，
仅改变 AdaRMS 条件投影的 autocast 行为。

BF16 第 11,763 步、普通 `t=0.1` 的分组结果如下。梯度指标指
`time_mlp_in.projector.weight`：

| FP32 保护范围                        |      loss | 梯度相对 L2 误差 | 梯度方向余弦 |
| ------------------------------------ | --------: | ---------------: | -----------: |
| 全 FP32 参照                         | 0.0767501 |                0 |            1 |
| 无额外保护                           | 0.0768491 |          1.22901 |      0.50210 |
| 仅 attention 前 AdaRMS               | 0.0768496 |          1.03035 |      0.59809 |
| 仅 FFN 前 AdaRMS                     | 0.0767245 |          0.21133 |      0.97752 |
| 仅最终 AdaRMS                        | 0.0768104 |          1.22754 |      0.50327 |
| 18 个 block 的两处 AdaRMS，36 个投影 | 0.0767468 |          0.02931 |      0.99958 |
| 所有 AdaRMS，含最终 norm，37 个投影  | 0.0767087 |          0.03234 |      0.99950 |

只保护前、中、后六个 block 时误差分别为 0.554、0.918、0.983。
因此没有证据支持只改某几个 block 的编号；应该保护这种条件投影模块。
FFN 前的投影贡献最大，但只保护它仍不足。最终 norm 影响较小，统一保护
它能维持同一种条件投影的精度约定，未观察到需要将整个 block 改为 FP32。

所有 AdaRMS 保护后的时间 MLP 输入投影梯度结果：

| checkpoint   | 条件       | 原 BF16 相对误差 | 修复后相对误差 | 修复后余弦 |
| ------------ | ---------- | ---------------: | -------------: | ---------: |
| BF16 epoch 1 | 普通 t=0.1 |          1.22901 |        0.03234 |    0.99950 |
| BF16 epoch 1 | RTC t=0.1  |          0.63675 |        0.02382 |    0.99972 |
| BF16 epoch 1 | RTC t=0.5  |          0.06754 |        0.01301 |    0.99992 |
| FP32 epoch 6 | 普通 t=0.1 |          0.22577 |        0.05377 |    0.99905 |
| FP32 epoch 6 | RTC t=0.1  |          0.28860 |        0.04540 |    0.99916 |
| FP32 epoch 6 | RTC t=0.5  |          0.13694 |        0.01949 |    0.99997 |

时间 MLP 输出投影也满足预定阈值。这说明问题是可复现的反向精度偏差：
`openpi_fp32_flow` 已保护时间 MLP，但下游的 `dense(cond)` 仍被 autocast
降为 BF16，回传的条件梯度因而偏离 FP32。将输出事后 `.float()` 无法
恢复矩阵运算中丢失的精度。loss 接近不代表梯度方向接近。
这些结果尚未证明它是长程退化的唯一原因。

## 实现范围

新增精度保护由 `model.llm_expert.adarms_fp32=True` 控制，保护：

```text
llm_expert.layers.0..17.input_layernorm.dense
llm_expert.layers.0..17.post_attention_layernorm.dense
llm_expert.norm.dense
```

`GemmaRMSNorm` 在这条路径上关闭 autocast，使用 FP32 输入和权重执行
`F.linear`，保留其 FP32 backward。默认开关为 False，不改变其他配置的
默认精度，也不增加 checkpoint 参数。
37 个投影有 116,505,600 个参数，占完整模型 3,616,778,010 参数的 3.22%。

已有 FP32 区域继续保留：时间 embedding/MLP、动作输入/输出投影、flow
目标和 loss、视觉 patch embedding，以及 FP32 master 下的 residual stream。
Residual 修正分别由 `model.preserve_fp32_residuals` 和
`model.vision_backbone.preserve_fp32_residuals` 控制，默认均为 False。
只有独立修复配置显式开启它们和 `adarms_fp32`，原配置不会隐式采用修复路径。
实际 hook 确认 SigLIP encoder FC、语言与动作 expert 的 Q projection、
expert FFN up projection 仍输出 BF16。新增保护仅作用于条件投影，没有
关闭整个 transformer 的 autocast。

主权重和 Adam 状态仍为 FP32，这是已有 `keep_params_fp32=True` 的行为。
本方案减少的是 FP32 矩阵计算，不宣称将参数和优化器显存减半。

### 推理精度配置

独立修复配置的 `inference_model` 已包含上述三个模型开关，`inference`
现也提供独立的推理精度默认值：

```python
inference = dict(
    enable_mixed_precision=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True)
```

`scripts/inference.py` / `scripts/inference_real_robot.py` 的 runner 读取
`inference`，不会读取训练的 `runner` 精度设置。`keep_params_fp32=True`
使其不对整个模型调用 `.to(dtype=torch.bfloat16)`，由 autocast 控制大矩阵
运算。机器人部署配置可继承这份配置，并补上 `inference.type`、dataset、
denormalizer、operator 等硬件及数据定义。这一小节本身不是完整的实机配置。
如果部署子配置再次覆盖精度字段，以最终合并后的配置为准。

## 验证与运行方式

22 项定向 pytest 通过，覆盖普通和 RTC 条件的梯度、实际 Gemma attention
及 FFN 仍用 BF16、PI0/PI05 forward/backward/inference 和核心对齐。
flake8、isort、`git diff --check` 通过。

使用两张 A800、最终 EMA checkpoint、microbatch 2/rank、RTC、AdamW、EMA，
通过原 FSDP runner 做连续更新检查。诊断使用新优化器和六个固定样本循环，
用于执行验证与计时，不能当成长程训练或泛化验证。

三条路径均在每个 rank 完成 24 次更新，loss、梯度范数均有限，Adam step
均为 24。排除前四次预热，取各 rank 的单步耗时中位数，再取较慢 rank：

| 模式                 | 单步秒数 | 每卡峰值 allocated GB |
| -------------------- | -------: | --------------------: |
| BF16，无 AdaRMS 保护 |   1.2342 |                47.771 |
| BF16，AdaRMS FP32    |   1.2369 |                47.780 |
| 全 FP32              |   1.9349 |                47.735 |

选择性保护相对未保护 BF16 的耗时差约 0.2%，在此次短测的波动范围内；
相对全 FP32 耗时约减少 36%，吞吐约为 1.56 倍。这是两卡小 batch 的
测量，不代表多机、batch 128/256 的端到端收益。优化器和主权重仍是 FP32，
本测量的显存主要由它们决定。三个诊断模式使用相同模型代码，只改变计算
精度与 AdaRMS 开关，未保护 BF16 的 residual 行为对应 `786f06`。

复训使用已更新工作区的独立修复配置：

```bash
bash scripts/train.sh \
  configs/pi05/pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune.py \
  ./work_dirs/pi05_basket_bf16_adarms_fp32 \
  --cfg-options train_dataloader.per_device_batch_size=8 \
  runner.max_epochs=6 runner.max_steps=None
```

独立配置已显式设置模型与 runner 的混合精度开关为 True、计算 dtype 为 BF16、
`keep_params_fp32=True`、`adarms_fp32=True`。动作仍为原生 42 维。
要直接对照旧 BF16 退化实验，保持相同的 16 个总 ranks、每 rank batch 8、
累积 1，并从同一 base checkpoint 初始化。不要再次 checkout 旧 SHA，
否则会回到没有这项保护的代码。

证据位于 `work_dirs/pi05_selective_precision_20260920/`：
`bf16_epoch1.json`、`fp32_epoch6.json` 包含全部分组和参数梯度，
`runs_summary.json` 包含完整日志汇总。探针脚本及 runner 输出也保存在该目录。
`throughput_summary.json` 汇总计时，`environment.json` 记录 PyTorch
2.8.0+cu128、Transformers 5.3.0、设备信息及源码/固定 batch 的 SHA256。
下一步的稳定性判据是在相同 batch 和数据设置下完成原退化区间之后的复训。

## 旧实验兼容性审查

隔离前，虽然 AdaRMS 默认关闭，两处 residual 修正仍可能影响其他启用
OpenPI 混合精度的旧配置。现在三个新开关都默认关闭，原 basket 主配置也
不再隐式启用 AdaRMS 修复。用户已有的数据路径修改保留，不属于本次数值修复。

在同一完整 42 维模型、最终 EMA 权重、两个真实样本、固定 noise/time/RTC
种子和确定性 GPU 设置下，从 `6dbfc8` 原源码提取受改动函数作参照：

| 对照                         | loss       | predictions | 全部 805 个有梯度的参数张量 |
| ---------------------------- | ---------- | ----------- | --------------------------- |
| 旧默认 FP32 vs 6dbfc8 原实现 | 逐元素相同 | 逐元素相同  | 逐元素相同                  |
| 旧默认 BF16 vs 6dbfc8 原实现 | 逐元素相同 | 逐元素相同  | 逐元素相同                  |
| 新修复 FP32 vs 隔离前修复    | 逐元素相同 | 逐元素相同  | 逐元素相同                  |
| 新修复 BF16 vs 隔离前修复    | 逐元素相同 | 逐元素相同  | 逐元素相同                  |

25 项定向 pytest 通过，包括原配置、FP32 配置、新修复配置的模型和
inference_model 开关隔离。兼容性证据位于
`work_dirs/pi05_legacy_compatibility_20260920/`。
该检查证明这次开关隔离保留了受测路径，不保证跨硬件、依赖版本或分布式
条件的完整训练逐位相同。其他历史 commit 若本身具有不同算法/精度逻辑，
应继续用其原始 commit 复现。
