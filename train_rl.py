import os

import ray
from ray import tune
from ray.tune.registry import register_env

from env import MyEnv

print(os.listdir())
print(os.listdir(".."))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def env_creator(env_config):
    return MyEnv("observations_df.csv", "assets_df.csv", n_assets=96)


CONFIG = {
    "num_workers": 15,
    "num_envs_per_worker": 1,
    "create_env_on_driver": False,
    "rollout_fragment_length": 512,
    "batch_mode": "truncate_episodes",
    "num_gpus": 1,
    "train_batch_size": 7680,
    "model": {
        "fcnet_hiddens": [2048, 1024, 512, 512],
        "fcnet_activation": "relu",
        "conv_filters": None,
        "conv_activation": "relu",
        "free_log_std": False,
        "no_final_linear": False,
        "vf_share_layers": True,
        "use_lstm": False,  # TODO: TURN TO TRUE
        "max_seq_len": 256,
        "lstm_cell_size": 256,
        "lstm_use_prev_action_reward": False,
        "_time_major": False,
        "framestack": True,
        "dim": 84,
        "grayscale": False,
        "zero_mean": True,
        "custom_model": None,
        "custom_model_config": {},
        "custom_action_dist": None,
        "custom_preprocessor": None,
    },
    "optimizer": {},
    "gamma": 0.99,
    "horizon": None,
    "soft_horizon": False,
    "no_done_at_end": False,
    "env_config": {},
    "env": "MyEnv",
    "normalize_actions": False,
    "clip_rewards": True,
    "clip_actions": True,
    "preprocessor_pref": "deepmind",
    "lr": 3e-04,
    "monitor": False,
    "log_level": "WARN",
    "ignore_worker_failures": True,
    "log_sys_usage": True,
    "fake_sampler": False,
    "framework": "tf",
    "eager_tracing": False,
    "explore": True,
    "exploration_config": {"type": "StochasticSampling"},
    "evaluation_interval": None,
    "evaluation_num_episodes": 10,
    "in_evaluation": False,
    "evaluation_config": {},
    "evaluation_num_workers": 0,
    "custom_eval_function": None,
    "sample_async": False,
    "_use_trajectory_view_api": True,
    "observation_filter": "NoFilter",
    "synchronize_filters": True,
    "tf_session_args": {
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {"allow_growth": True},
        "log_device_placement": False,
        "device_count": {"CPU": 1},
        "allow_soft_placement": True,
    },
    "local_tf_session_args": {
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
    "compress_observations": False,
    "collect_metrics_timeout": 180,
    "metrics_smoothing_episodes": 100,
    "remote_worker_envs": False,
    "remote_env_batch_wait_ms": 0,
    "min_iter_time_s": 0,
    "timesteps_per_iteration": 0,
    "seed": None,
    "extra_python_environs_for_driver": {},
    "extra_python_environs_for_worker": {},
    "num_cpus_per_worker": 1,
    "num_gpus_per_worker": 0,
    "custom_resources_per_worker": {},
    "num_cpus_for_driver": 1,
    "memory": 0,
    "object_store_memory": 0,
    "memory_per_worker": 0,
    "object_store_memory_per_worker": 0,
    "input": "sampler",
    "input_evaluation": ["is", "wis"],
    "postprocess_inputs": False,
    "shuffle_buffer_size": 0,
    "output": None,
    "output_compress_columns": ["obs", "new_obs"],
    "output_max_file_size": 67108864,
    "multiagent": {
        "policies": {},
        "policy_mapping_fn": None,
        "policies_to_train": None,
        "observation_fn": None,
        "replay_mode": "independent",
    },
    "logger_config": None,
    "replay_sequence_length": 1,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.95,
    "kl_coeff": 0.2,
    "sgd_minibatch_size": 256,
    "shuffle_sequences": True,
    "num_sgd_iter": 10,
    "lr_schedule": [[0, 3e-4], [1000000, 3e-5], [10000000, 3e-6], [100000000, 3e-8]],
    "vf_share_layers": False,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,
    "clip_param": 0.3,
    "vf_clip_param": 3000,
    "grad_clip": 1.0,
    "kl_target": 0.01,
    "simple_optimizer": False,
    "_fake_gpus": False,
}

if __name__ == "__main__":
    ray.init(num_gpus=1)
    register_env("MyEnv", env_creator)
    analysis = tune.run(
        "PPO",
        name="PPO_SpainAI_1002",
        stop={"episode_reward_mean": 10000},
        config=CONFIG,
        checkpoint_at_end=True,
        checkpoint_freq=1,
    )
