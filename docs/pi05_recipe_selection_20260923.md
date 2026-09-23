# PI05：ada0381668 实际运行仍选择旧配置

本次核对用户报告的
`pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune_ada0381668_bs256`。
已读取修正后的实验目录：

```text
/mnt/data/oss-wlcb/users/liyinhao/projects/FluxVLA/work_dirs/pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune_ada0381668_bs256
```

## 已确认的配置问题

用户命令 checkout `ada0381668`，但传给 `scripts/train.sh` 的文件仍是
`pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune.py`。
该 commit 为保留历史实验行为，没有改变这个配置的数值设置；新增方案位于
`pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py`。
只更新 commit 不会切换配置。

保存的 `config.json` 和 `run-metrics.jsonl` 中的实际启动参数确认，本次运行
确实选择旧配置，`resume_from=None`。用 MMEngine 解析当前旧配置并应用
用户的 batch/epoch/step 覆盖参数后，除新增提醒字段及统计文件路径外，
结果与保存配置完全相等。以下同时核对了保存配置和新配置的解析结果：

| 实际生效设置                   | 本次保存的 adarms 配置 | expert6 配置     |
| ------------------------------ | ---------------------- | ---------------- |
| expert AdaRMS FP32             | True                   | True             |
| backbone `attention_math_fp32` | False（默认）          | True             |
| expert `attention_math_fp32`   | False（默认）          | True             |
| expert `fp32_layers`           | 空（默认）             | 0, 1, 2, 3, 4, 5 |
| action/loss 维度               | 42                     | 42               |
| 每卡 batch                     | 8                      | 8                |
| `max_epochs` / `max_steps`     | 6 / None               | 6 / None         |
| `max_keep_ckpts`               | 2                      | 2                |

因此，这次实际运行没有启用上一轮新增的 attention 和 early-expert 保护。
它仍选择此前已观察到长程退化的 AdaRMS-only 路径。不能把这次反馈视作
expert6 方案已经经过长跑且失败的证据。

## 实际 loss 与历史对照

本次读取到连续的 13,302 条训练记录，步数为 1–13,302，无损坏行及
NaN/Inf loss。首步 raw loss 为 1.262554，最后一步的 100 步平滑 loss 为
0.206682，最后 1,000 步 raw loss 均值为 0.188980。

与 `9fa2cefebe` 的 AdaRMS 实验比较，两个保存配置的唯一差异是
`dataset_statistics_path`；统计 JSON 内容也完全相同。三次实验的统计
JSON 经键排序后的 SHA256 均为
`e5565f791cee0c48b5221912a5256abfae1ce8c381a6ef4e0c2d2857f28d518d`。
数据、RTC、优化器、LR 调度、42 维 action/loss、每卡 batch 8 及
梯度累积 1 均相同。下表使用同一步数区间的 **raw loss 算术均值**：

| 步数范围      | 本次 ada0381 + 旧配置 | 上次 9fa2cef + 旧配置 | 6dbfc8e 全 FP32 |
| ------------- | --------------------: | --------------------: | --------------: |
| 10,001–11,000 |              0.019555 |              0.019381 |        0.019129 |
| 11,101–11,200 |              0.038285 |              0.033644 |        0.019921 |
| 11,301–11,400 |              0.106053 |              0.094947 |        0.019312 |
| 11,501–11,600 |              0.151082 |              0.151859 |        0.017315 |

本次在第 11,120 步的平滑 loss 超过 0.03，上次为第 11,147 步；二者都在
约 11,100 步开始持续退化，FP32 对照保持约 0.02。本次第 13,001–13,302
步 raw loss 均值进一步升至 0.213197，同区间 FP32 为 0.016407。
这里是有限 loss 持续退化，不能从这些指标断言进程发生异常退出。

两次退化都早于第 11,765 步的 epoch 切换；该区间记录的 LR 约为
2.43e-5，没有 LR 跳升。与 FP32 对照相比，除精度设置、统计文件路径及
checkpoint 保留数量外，未发现其他配置差异。因此本次没有证据要求修改
relative action、loss 定义、学习率或 runner。

这些结果确认旧方案的退化重复发生。上一轮在坏训练 checkpoint 上隔离出的
融合 SDPA 反向误差及 early-expert 精度问题仍是新方案针对的数值缺陷；
不能仅凭同样的 loss 曲线证明本次每一步的内部故障完全相同，也不能将
这些缺陷认定为所有长程退化的唯一诱因。

## 本次修正

- 将旧配置标注为供历史复现的配置，并增加启动提醒，给出对应的新配置路径。
- `scripts/train.py` 在数据统计及数据集构建之前显示配置提供的提醒，仅 rank 0
  输出。expert6 子配置清除父配置的旧方案提醒。
- 比较变更前后解析结果，除提醒字段外，所有配置值完全相同；未修改模型数值
  实现、runner、batch 机制、学习率或历史配置的精度默认值。

正确训练入口如下，使用新的输出目录，从配置中的原 base checkpoint 开始：

```bash
bash scripts/train.sh \
  configs/pi05/pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py \
  ./work_dirs/pi05_basket_rtc_bf16_expert6_fp32_bs256_20260923 \
  --cfg-options train_dataloader.per_device_batch_size=8 \
  runner.max_epochs=6 runner.max_steps=None
```

环境变量的 `export` 命令应独占一行；用户粘贴的末行将 `export` 和
`bash scripts/train.sh` 连在了一起，应补上换行。

新配置在训练及 `inference_model` 中均启用两个 Gemma 分支的
`attention_math_fp32=True`，并将 expert 第 0–5 层的投影和 FFN 设为
FP32。其余 transformer 投影/FFN 继续 BF16 autocast，原有 AdaRMS、
flow、视觉 stem 和残差 FP32 保护继续保留；master 参数及 Adam 状态原本
就采用 FP32。没有把全部计算切换成 FP32。

## 证据与限制

13 项已有 attention 精度/配置/KV cache 测试通过，并通过启动提醒的定向
执行检查：旧配置在数据构建前输出提醒，新配置不输出旧方案提醒。
本次追加检查确认保存配置与旧配置一致，新配置的训练及推理保护设置
一致，同时保持 42 维、每卡 batch 8、六个 epoch 和最多保留 2 个 checkpoint。

原始解析结果、兼容性检查及本次实际日志摘要位于
`work_dirs/pi05_recipe_audit_20260923/`：

- `actual_run_audit.json`：实际配置差异、统计文件校验、区间 loss 和日志边界。
- `actual_run_loss_rows.json`：本次读取的三份训练日志快照。
- `actual_run_recipe_checks.json`：保存配置等价性及训练/推理精度开关核对。
- `loss_comparison.png` / `loss_comparison.svg`：10,000–13,302 步的对照曲线。

读取时新目录只包含第 11,764 步的 `.safetensors` checkpoint（EMA 权重），
没有包含训练权重和 Adam 状态的 `.pt` 文件；本次未对它做恢复训练或
重新测量内部 Q/K 梯度。runner 用保存间隔内累计 loss 的均值命名文件，
所以文件名中的 `loss=0.0320` 不是对应时刻的最新 loss：第 11,764 步
raw loss 为 0.058323，100 步平滑值为 0.149986。判断退化应依据逐步日志。

expert6 的既有数值证据和短程验证见
[attention 精度诊断](pi05_attention_instability_20260920.md)。其原数据、原有效
batch 上的完整六个 epoch 收敛仍未验证，不能承诺更换入口后一定不再退化。
