# DreamZero RoboCasa：从预训练底座开始微调

新配置 `configs/dreamzero/dreamzero_robocasa_full_data_joint_delta_lora_uniform_finetune.py`
直接继承原 full-data 配置，只设置 `model.vla_head.action_timestep_weighting='uniform'`。
从原始 AgiBot 预训练权重初始化新的 LoRA 和优化器，不加载之前训练出的 RoboCasa 权重。

训练参数沿用原方案：全部 24 个任务、1–4 个动作块、rank-4 LoRA、AdamW、
峰值 LR 1e-5、100,000 次更新和 5% warmup。视频目标与噪声采样分布保持原样，
只把动作时间步损失权重改为 1。DDP runner 使用原实现。

在仓库根目录激活 `fluxvla` 环境后，使用本机的两张 GPU、新输出目录启动：

```bash
export DREAMZERO_CKPT_ROOT=/mnt/data/cpfs/mnt/data/liyinhao/checkpoints
export ROBOCASA_DATA_ROOT=/root/projects/FluxVLA/datasets/robocasa_lerobot_V2.1_fixed
export PYTHONHASHSEED=7
export NPROC_PER_NODE=2 WORLD_SIZE=1 RANK=0

bash scripts/train.sh \
  configs/dreamzero/dreamzero_robocasa_full_data_joint_delta_lora_uniform_finetune.py \
  work_dirs/dreamzero_robocasa_uniform_bs64_fresh \
  --cfg-options runner.grad_accumulation_steps=16
```

此命令的 global batch 是 `2 GPU × 2 samples × 16 accumulation = 64`。
16 GPU 时将累积次数改为 2，32 GPU 时改为 1。启动脚本的 `WORLD_SIZE` 表示节点数，
多机按原脚本约定设置节点 rank、master 地址和端口。
新配置继承原配置的累积次数 1，启动时按 GPU 数显式覆盖即可。

不需要适配器路径或 `--resume-from`。用原来的
`dreamzero_robocasa_full_data_joint_delta_lora_finetune.py` 加相同 batch 覆盖，即为原目标对照。
也可以在新配置的命令末尾加 `model.vla_head.action_timestep_weighting=scheduler` 做对照。

保留已有的动作采样诊断与评测可复现性修复。默认每 1,000 步记录训练样本上的
`sampled_action_mae_first8/first24` 等指标；它们不是仿真成功率。
均匀权重改变了 action loss 的标尺，两种目标的 loss 数字不能直接用于判断优劣。
该权重方案仍需完整 RoboCasa 评测验证，尚未证明成功率超过 12%。

此前短程实验的数值和限制见 [历史实验报告](dreamzero_robocasa_continuation_20261027.md)。
