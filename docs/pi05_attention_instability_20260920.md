# PI05：9fa2cef 后续退化与 attention 精度诊断

## 结论与边界

`9fa2cefebe` 的 AdaRMS FP32 修正不足以消除长程退化。本次确认了另一处
可复现的数值问题：expert 早期层的 Q/K 激活变大后，融合 SDPA 的反向计算
明显偏离 FP64 参考；**只把 SDPA 输入转为 FP32 仍不能排除该误差**。
显式选择 FP32 math attention 可以消除这部分内核误差。继续保护 expert
前六层的投影和 FFN，可显著减小完整模型的梯度偏差。

这说明存在需要修正的 attention 数值误差，但尚未证明它是最初触发退化的
唯一原因。退化前后只有 epoch checkpoint，无法观察第 11,000 步附近每次
更新的内部状态。以下短程验证也**不构成新方案六个 epoch 不退化的保证**。
目前完成六个 epoch 的稳定对照仍是 `6dbfc8edb0ed` 的全 FP32 实验。

## 同 batch 的长跑对照

对照运行：

- 失败：`pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune_9fa2cefebe_bs256`。
- 稳定：`pi05_paligemma_basket_all_rtc_fp32_full_finetune_6dbfc8edb0ed_bs256`。

保存配置确认两次均为有效 batch 256、每卡 batch 8、每 epoch 5,882 步，
native action/loss 42 维、horizon 32、相同 RTC 设置。优化器、学习率调度、
梯度裁剪和数据配置相同，`dataset_statistics.json` 内容完全相等。
除精度设置、统计文件路径和 checkpoint 保留数量外，没有发现配置差异。
未证明分布式随机数流和逐步数据顺序完全相同。

| 步数范围      | 9fa2cef raw loss 均值 | FP32 raw loss 均值 |
| ------------- | --------------------: | -----------------: |
| 9,001–10,000  |              0.019915 |           0.020003 |
| 10,001–11,000 |              0.019381 |           0.019129 |
| 11,101–11,200 |              0.033644 |           0.019921 |
| 11,301–11,400 |              0.094947 |           0.019312 |
| 11,501–11,600 |              0.151859 |           0.017315 |
| 11,601–11,689 |              0.178050 |           0.018723 |

日志没有 NaN/Inf；这是有限 loss 持续退化，起点不在 epoch 切换处。
两份第 11,764 步 checkpoint 都有 FP32 master 参数和 FP32 Adam 一、二阶状态。

固定两个真实样本、noise、RTC 随机种子，取 `t=0.1`，使用 `train_model`
而非 EMA 权重：坏 checkpoint 即使用 FP32 计算，loss 也为 0.987002；稳定
FP32 的同一步 checkpoint 为 0.054335。因此切换推理精度不能修复坏权重。

## 从真实 attention 输入隔离内核误差

取坏 checkpoint 第 11,764 步的完整前向输入和反向 cotangent，将每层的
Q/K/V 单独重新计算，使用 FP64 math SDPA 作参考。第 2 层（索引 1）的
Q 最大绝对值约 4,409；同一运行第 5,882 步该层约为 16.3。

| 第 2 层计算路径 | Q 梯度相对 L2 误差 | K 梯度相对 L2 误差 |
| --------------- | -----------------: | -----------------: |
| BF16 默认 SDPA  |            18.3273 |            33.7603 |
| FP32 默认 SDPA  |            0.24465 |            0.26268 |
| FP32 math SDPA  |        约 0.000001 |        约 0.000001 |

表中 BF16 一项包含输入舍入和内核误差。另做独立合成实验，先把输入统一
舍入到 BF16 可表示值，再分别在 FP64/FP32/BF16 计算，从而排除输入不一致：
大 Q 情况下，BF16 fused attention 的 Q/K 梯度相对误差仍约为 1.22/1.36，
FP32 math 为约 0.000103/0.000102。因此误差不只来自 checkpoint 或输入转换。

融合 FP32 SDPA 在坏权重上也有误差，所以后续完整模型消融改用
**全 FP32 + math attention** 为参考，避免用不稳定的融合内核作为真值。

## 完整模型消融及实现核对

同样使用第 11,764 步 `train_model`、固定两个样本、RTC、`t=0.1`。
全模型指标为拼接全部有效参数梯度后的相对 L2 误差，不是逐层误差均值。

| 路径                                 |     loss | 全模型梯度相对误差 | timestep 输入 MLP 梯度相对误差 |
| ------------------------------------ | -------: | -----------------: | -----------------------------: |
| 全 FP32 + math attention             | 0.987001 |                  0 |                              0 |
| 9fa2cef AdaRMS 修正                  | 0.998314 |            0.46479 |                        0.39513 |
| 仅再保护 math attention              | 0.999198 |            0.26364 |                        0.17727 |
| math attention + expert 前 4 层 FP32 | 0.987894 |            0.05584 |                        0.02701 |
| math attention + expert 前 6 层 FP32 | 0.986957 |            0.03512 |                        0.01394 |
| math attention + 全部 expert FP32    | 0.987100 |            0.00232 |                        0.00081 |

新增实现与消融脚本的前六层方案，loss、逐参数梯度统计完全相同。
选择前六层是当前样本上的精度与计算量折中；这些测量不证明其他样本、
其他 checkpoint 或后续训练永远只需保护这六层。

## 配置和兼容性

新增独立配置：

```text
configs/pi05/pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py
```

继承原 AdaRMS 配置，新增：

```python
model = dict(
    llm_backbone=dict(attention_math_fp32=True),
    llm_expert=dict(
        attention_math_fp32=True, fp32_layers=(0, 1, 2, 3, 4, 5)))
inference_model = model.copy()
```

| 计算部分                                                            | 新配置精度              |
| ------------------------------------------------------------------- | ----------------------- |
| expert 第 0–5 层 Q/K/V/O 投影及 FFN                                 | FP32                    |
| expert 第 6–17 层投影及 FFN                                         | BF16 autocast           |
| 两个 Gemma 分支的 attention 核心、softmax 及反向                    | FP32，显式 math backend |
| 视觉 encoder、语言主干的投影及 FFN                                  | BF16 autocast           |
| 所有 37 个 expert AdaRMS、时间 MLP、flow 投影/目标、视觉 stem、残差 | 延续原 FP32 保护        |
| master 参数、Adam 状态                                              | 全部 FP32，延续原设置   |

前六层与全部 AdaRMS 的参数集合去重后为 220,314,624，占完整模型参数
3,616,778,010 的 6.09%。该比例描述这些受保护模块，**不是 FP32 参数存储比例**；
attention 核心自身没有额外参数，flow/stem 等已有 FP32 模块另计。

新开关默认关闭，不增加或重命名 checkpoint 参数。原 basket、全 FP32 和
9fa2cef AdaRMS 配置的设置不变；native 42 维、runner、优化器、学习率和
batch 机制不变。新配置继承 `max_keep_ckpts=2`。推理配置同步继承精度设置，
仍需部署配置提供 runner、dataset、denormalizer 和 operator。

用 `git show 9fa2cef` 提取旧方法，对小型双分支模型的 joint 和普通 Gemma
路径比较：FP32/BF16 下输出及所有有效梯度逐元素相等。这是这些受测路径
的默认兼容性证据，不是跨硬件完整随机训练逐位一致的承诺。
新路径还检查了带 KV cache 的 suffix 与整段 block mask 计算一致。

## 验证与局限

- 45 项定向回归测试通过，覆盖大 logit 的 FP64 梯度对照、选择性精度、
  KV cache、隐式 causal mask、原配置、动作 horizon 和时间采样。
- 恢复坏 checkpoint 的 `train_model` 和第 11,764 步 Adam 状态，原路径与
  新路径各训练 128 步。使用 6 个固定真实样本、batch 2、每步重新采样
  noise/time、原 RTC 和固定保存 LR；两条路径 loss/梯度均有限。
  最后 32 步均值分别 0.44094/0.40131，不能作为已恢复原任务性能的证据。
  此压力测试不使用 EMA，也没有复现完整数据或原 global batch。
- 两卡完整 FSDP runner、AdamW、EMA、RTC、每卡 batch 2：新路径 16 步、
  9fa2cef 对照 12 步均通过。当前环境中去掉前四步的中位时间约为
  3.86/3.53 秒，额外开销约 9%；峰值 allocated memory 约 47.9/47.8 GB。
  这是本机短测，不能推断用户 32 卡集群的吞吐。
- 每卡 batch 8 的双卡完整 FSDP、AdamW、EMA 另通过 6 次更新；使用重复样本
  验证实际 batch 形状与执行内存，峰值 allocated memory 约 70.2 GB。
- FP32 math attention 会显式分配 attention 矩阵，长序列的显存开销与融合
  SDPA 不同，不能把默认配置推广到任意分辨率或序列长度。
- 未重跑原 global batch 256 的六个 epoch。新方案需要穿过此前约
  11,000 步的退化区间并完成长跑，才能判断是否解决最终训练退化。
- 扩展检查中的一个原有未跟踪 RoboCasa 测试引用了不存在的配置文件，
  因 `FileNotFoundError` 失败；没有修改该无关测试或补造它依赖的配置。

建议从原 base checkpoint 开始，使用新目录：

```bash
bash scripts/train.sh \
  configs/pi05/pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py \
  ./work_dirs/pi05_basket_rtc_bf16_expert6_fp32_bs256 \
  --cfg-options train_dataloader.per_device_batch_size=8 \
  runner.max_epochs=6 runner.max_steps=None
```

复现脚本和原始 JSON 保存在 `work_dirs/pi05_remaining_precision_20260920/`：
`gradient_probe.py`、`attention_probe.py`、`implementation_epoch2.json`、
`attention_epoch1.json`、`attention_epoch2.json`、`checkpoint_stats.json`、
`resume_probe.py`、`resume_old.json`、`resume_new.json`、`training_probe.py`、
`training_control.py`、`legacy_parity.json` 及测试日志。
独立合成内核对照见 `synthetic_attention_probe.py` 和 `synthetic_attention.json`。
环境：PyTorch 2.8.0+cu128、Transformers 5.3.0、两张 A800 80GB。
