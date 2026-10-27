# DreamZero RoboCasa 多任务续训

这两套配置把 2026-10-27 的短程实验扩展到全部 24 个任务。
它们是待验证的训练方案，尚未证明能把完整评测成功率提升到 12% 以上。

| 配置后缀              | 动作时间步损失权重       | 视频损失权重 |
| --------------------- | ------------------------ | ------------ |
| `continue_control.py` | 原 scheduler 权重        | 1            |
| `continue_uniform.py` | 每个已采样时间步权重为 1 | 1            |

两组使用相同原始 AgiBot 底座和原 100k LoRA/投影层参数。保留原有全部任务、
图像增强、1–4 个动作块、每块 24 步、rank-4 LoRA 和 FP32 训练参数。
新 AdamW 优化器运行 2,000 次更新，峰值 LR 1e-6，前 100 步 warmup，之后 cosine decay。
每 250 步记录动作误差，每 500 步保存 checkpoint，保留四个。

## 启动

在更新后的 FluxVLA 仓库根目录、激活 `fluxvla` 环境后运行。
本机已有的原 100k 适配器快照可直接使用：

```bash
export DREAMZERO_CONTINUE_ADAPTER=/mnt/data/cpfs/mnt/data/liyinhao/projects/fluxvla/dreamzero_fixes/20261027_continuation_pilot/initial_adapter
export DREAMZERO_CKPT_ROOT=/mnt/data/cpfs/mnt/data/liyinhao/checkpoints
export ROBOCASA_DATA_ROOT=/root/projects/FluxVLA/datasets/robocasa_lerobot_V2.1_fixed
export PYTHONHASHSEED=7
export NPROC_PER_NODE=2 WORLD_SIZE=1 RANK=0

bash scripts/train.sh \
  configs/dreamzero/dreamzero_robocasa_joint_delta_lora_continue_control.py \
  work_dirs/dreamzero_robocasa_continue_control_20261027

bash scripts/train.sh \
  configs/dreamzero/dreamzero_robocasa_joint_delta_lora_continue_uniform.py \
  work_dirs/dreamzero_robocasa_continue_uniform_20261027
```

使用两个新的输出目录。也可以将 `DREAMZERO_CONTINUE_ADAPTER` 指向包含
`adapter_config.json` 和 `adapter_model.safetensors` 的原 100k 训练目录。
不指定该变量时，配置默认使用原训练目录在 `work_dirs/` 下的相对路径。
不要将单任务 pilot 的 200 步适配器作为这次对照的起点。

这是加载旧权重后重新开始优化器与学习率计划。**不要传 `--resume-from`，也不要把
`model.pretrained_name_or_path` 改为已合并的 RoboCasa checkpoint。**
加载器会检查 LoRA 配置、全部适配器/投影层键与形状，避免静默漏载。
本轮新 checkpoint 的 step 以 0 为起点，不代表从头训练。

启动脚本的 `WORLD_SIZE` 表示节点数；torchrun 传入训练进程的 `WORLD_SIZE` 表示总 GPU 数。
配置按后者计算梯度累积，保持 `2 × 总 GPU 数 × 累积次数 = 64`：
2 GPU 累积 16 次，16 GPU 累积 2 次，32 GPU 累积 1 次。
多机按现有启动约定设置节点数、节点 rank、master 地址与端口。
若手动覆盖每 GPU batch 或累积次数，需要同时调整以保持比较条件一致。

## 监控与选择

保留原来的 `eval/sampled_action_mae_first8`、`first24` 及对应 RMSE，新增：

- `eval/sampled_arms_command_mae_first8`：手臂实际指令误差，弧度。
- `eval/sampled_hands_command_mae_first8`：手部实际指令误差，控制器命令单位。
- `eval/sampled_waist_command_mae_first8`：腰部实际指令误差，弧度。
- 各组对应的 `command_rmse`、`active_normalized_mae`、`active_normalized_rmse`，
  以及 `first24` 指标。

指令误差在归一化预测限幅到 \[-1, 1\] 后，使用每个样本随数据传入的 min/max 计算。
统计量与动作均使用 N15 顺序，排除时间和维度 padding。
常量坐标的实际指令误差为零；`active_normalized` 指标还从分母中排除常量坐标，
因此右手常量坐标 25 不会制造无意义的退化。原始归一化指标保留以便连续对照。

默认训练 evaluator 每个 rank 抽取一个训练样本，仅用首帧、首状态和条件提示采样。
这些曲线是训练诊断，**不是独立验证集或仿真成功率**。可以按现有入口配置
`val_dataloader.dataset` 提供独立数据，但它必须有相同的动作统计与预处理契约。
均匀权重改变了训练 action loss 的标尺，不应直接比较两种目标的 loss 大小。

选择 checkpoint 时使用相同场景的 RoboCasa 闭环评测，固定启动 hash seed 与 checkpoint 路径，
保留初始观测指纹。两套配置继承 24 任务 × 50 回合、每次执行 8 步的原评测设置。
具体 pilot 的结果与限制见 [实验报告](dreamzero_robocasa_continuation_20261027.md)。
