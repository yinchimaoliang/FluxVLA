"""Simulator-only RoboDojo configuration for the FluxThemis ZMQ client.

This does not load a policy or checkpoint locally. Start a compatible stateless
FluxVLA model service separately, with evaluation reporting enabled, and pass
its public address to FluxThemis with ``--server-endpoint tcp://HOST:15555``.

Benchmark semantics match pi05_paligemma_robodojo_full_finetune.py; the old
themis.runner / model_client / ros_server interfaces are intentionally absent.
"""

_GENERALIZATION_BASE_TASKS = (
    'stack_bowls',
    'push_T',
    'pack_objects_into_box',
    'fold_clothes',
    'hang_mugs',
    'sweep_blocks',
    'pour_liquid_into_cup',
    'make_toast',
    'arrange_largest_number',
    'sort_nesting_dolls_by_size',
    'store_laptop_and_headphones',
    'stack_blocks',
)

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key='robodojo_arx_x5',
    ),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            action_mode='joint',
            headless=True,
        ),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides={
            task_name: 25
            for base_task in _GENERALIZATION_BASE_TASKS
            for task_name in (base_task, f'{base_task}_random')
        },
        max_episode_steps=2000,
        execute_horizon=50,
        stop_on_success=True,
    ),
)
