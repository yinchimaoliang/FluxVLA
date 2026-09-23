# PI05：ada0381668 运行命令仍选择旧配置

本次核对用户报告的
`pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune_ada0381668_bs256`。

## 已确认的配置问题

用户命令 checkout `ada0381668`，但传给 `scripts/train.sh` 的文件仍是
`pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune.py`。
该 commit 为保留历史实验行为，没有改变这个配置的数值设置；新增方案位于
`pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py`。
只更新 commit 不会切换配置。

用 MMEngine 加载两个配置，并应用用户的 batch/epoch/step 覆盖参数，结果如下：

| 实际生效设置                   | 用户命令中的 adarms 配置 | expert6 配置     |
| ------------------------------ | ------------------------ | ---------------- |
| expert AdaRMS FP32             | True                     | True             |
| backbone `attention_math_fp32` | False（默认）            | True             |
| expert `attention_math_fp32`   | False（默认）            | True             |
| expert `fp32_layers`           | 空（默认）               | 0, 1, 2, 3, 4, 5 |
| action/loss 维度               | 42                       | 42               |
| 每卡 batch                     | 8                        | 8                |
| `max_epochs` / `max_steps`     | 6 / None                 | 6 / None         |
| `max_keep_ckpts`               | 2                        | 2                |

因此，这条命令没有启用上一轮新增的 attention 和 early-expert 保护。
它仍选择此前已观察到长程退化的 AdaRMS-only 路径。不能把这次反馈视作
expert6 方案已经经过长跑且失败的证据。

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

## 证据与限制

本环境中用户给出的 OSS 大写 `projects/FluxVLA` 路径不存在；小写
`projects/fluxvla/work_dirs` 没有该实验；CPFS 的同名实验目录为空。
因此本次没有读取到新的训练日志、保存配置或 checkpoint，不能判断本次
退化的步数、幅度及是否伴随其他问题。上述结论依据提交源码和用户命令的
配置解析结果，不是依据本次运行的保存配置。

13 项已有 attention 精度/配置/KV cache 测试通过，并通过启动提醒的定向
执行检查：旧配置在数据构建前输出提醒，新配置不输出旧方案提醒。
原始解析结果与兼容性检查位于
`work_dirs/pi05_recipe_audit_20260923/`。

expert6 的既有数值证据和短程验证见
[attention 精度诊断](pi05_attention_instability_20260920.md)。其原数据、原有效
batch 上的完整六个 epoch 收敛仍未验证，不能承诺更换入口后一定不再退化。
