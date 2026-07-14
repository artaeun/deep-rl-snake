# ppo training script for snake

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from source.gym_environments import gym_SnakeEnvironment


# configuration
OUTPUT_NAME = None
SAVE_DIR = 'weights'
TIMESTEPS = 3_000_000
RESUME_PATH = None
DEVICE = 'auto'
BOARD_SIZE = 10
MASK_SIZE = None
N_ENVS = 16
REWARD_PRESET = 'default_rewards'
SEED = 0
LEARNING_RATE = 3e-4
GAMMA = 0.99
ENT_COEF = 0.01


# reward presets as defined in TODO.md
REWARD_PRESETS = {
    'default_rewards': {
        'win': 1.0,
        'fruit': 0.5,
        'step': 0.0,
        'ate_himself': -0.2,
        'hit_wall': -0.1,
    },
    'high_penalty': {
        'win': 10.0,
        'fruit': 1.0,
        'step': 0.0,
        'ate_himself': -1.0,
        'hit_wall': -1.0,
    },
    'step_penalty': {
        'win': 1.0,
        'fruit': 0.5,
        'step': -0.001,
        'ate_himself': -0.2,
        'hit_wall': -0.1,
    },
}


class SnakeCNN(BaseFeaturesExtractor):
    # custom cnn for snake observations

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # get input channels from observation space (H, W, C format)
        input_channels = observation_space.shape[2]

        # build CNN layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        # calculate flattened size by running a dummy input
        dummy_input = torch.zeros(1, input_channels,
                                   observation_space.shape[0],
                                   observation_space.shape[1])
        flattened_size = self._get_flattened_size(dummy_input)

        # final linear layer
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, features_dim),
            torch.nn.ReLU(),
        )

    def _get_flattened_size(self, dummy_input):
        # calculates output size
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
            flattened_size = output.shape[1]
            return flattened_size

    def forward(self, observations):
        # forward pass
        # convert from (B, H, W, C) to (B, C, H, W) for Conv2d
        x = observations.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.linear_layer(x)
        return x


class MetricsCallback(BaseCallback):
    # callback to track training metrics

    def __init__(self, log_frequency=1000, verbose=1):
        super().__init__(verbose)
        self.log_frequency = log_frequency

        # storage for all episode data
        self.episode_rewards = []
        self.episode_lengths = []

        # storage for periodic logging
        self.logged_timesteps = []
        self.logged_mean_rewards = []
        self.logged_mean_lengths = []

    def _on_step(self):
        # called after each step
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

        # log to terminal periodically
        should_log = (self.n_calls % self.log_frequency == 0)
        has_episodes = (len(self.episode_rewards) > 0)

        if should_log and has_episodes:
            self._log_progress()

        return True

    def _log_progress(self):
        # prints progress to console
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]

        mean_reward = np.mean(recent_rewards)
        mean_length = np.mean(recent_lengths)

        # store for later plotting
        self.logged_timesteps.append(self.num_timesteps)
        self.logged_mean_rewards.append(mean_reward)
        self.logged_mean_lengths.append(mean_length)

        # print progress
        if self.verbose >= 1:
            total_episodes = len(self.episode_rewards)
            print(f"[Step {self.num_timesteps:>8,}] Mean Reward: {mean_reward:>8.2f} | Mean Length: {mean_length:>6.1f} | Episodes: {total_episodes:,}")

    def get_metrics(self):
        # returns collected metrics
        metrics = {}
        metrics['episode_rewards'] = self.episode_rewards.copy()
        metrics['episode_lengths'] = self.episode_lengths.copy()
        metrics['logged_timesteps'] = self.logged_timesteps.copy()
        metrics['logged_mean_rewards'] = self.logged_mean_rewards.copy()
        metrics['logged_mean_lengths'] = self.logged_mean_lengths.copy()
        return metrics


class SingleCheckpointCallback(BaseCallback):
    # callback that keeps only one checkpoint at a time

    def __init__(self, save_freq, save_path, name_prefix, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.current_checkpoint_path = None

    def _on_step(self):
        # called after each step
        should_save = (self.n_calls % self.save_freq == 0)

        if should_save:
            self._save_checkpoint()

        return True

    def _save_checkpoint(self):
        # saves new checkpoint and deletes old one

        # delete previous checkpoint if it exists
        if self.current_checkpoint_path:
            if os.path.exists(self.current_checkpoint_path):
                os.remove(self.current_checkpoint_path)
                if self.verbose >= 1:
                    print(f"Deleted old checkpoint: {self.current_checkpoint_path}")

        # create new checkpoint path
        checkpoint_name = f"{self.name_prefix}_checkpoint.zip"
        new_path = os.path.join(self.save_path, checkpoint_name)

        # save the model
        self.model.save(new_path)
        self.current_checkpoint_path = new_path

        if self.verbose >= 1:
            print(f"Saved checkpoint: {new_path}")

    def get_checkpoint_path(self):
        # returns current checkpoint path
        path = self.current_checkpoint_path
        return path


def build_config():
    # builds config dict from configuration variables
    if REWARD_PRESET in REWARD_PRESETS:
        rewards = REWARD_PRESETS[REWARD_PRESET]
    else:
        print(f"Warning: Unknown preset '{REWARD_PRESET}', using 'default_rewards'")
        rewards = REWARD_PRESETS['default_rewards']

    # determine observability mode
    if MASK_SIZE:
        if MASK_SIZE > 0:
            mask_size = MASK_SIZE
        else:
            mask_size = None
    else:
        mask_size = None

    # build config step by step
    config = {}

    # environment settings
    config['board_size'] = BOARD_SIZE
    config['mask_size'] = mask_size
    config['n_parallel_envs'] = N_ENVS
    config['rewards'] = rewards
    config['reward_preset_name'] = REWARD_PRESET

    # training settings
    config['total_timesteps'] = TIMESTEPS
    config['seed'] = SEED

    # ppo hyperparameters
    config['learning_rate'] = LEARNING_RATE
    config['n_steps'] = 1024
    config['batch_size'] = 2048
    config['n_epochs'] = 15
    config['gamma'] = GAMMA
    config['gae_lambda'] = 0.9
    config['clip_range'] = 0.2
    config['clip_range_vf'] = None
    config['normalize_advantage'] = True
    config['ent_coef'] = ENT_COEF
    config['vf_coef'] = 0.5
    config['max_grad_norm'] = 1.0

    # network architecture
    config['features_dim'] = 256
    config['policy_hidden_layers'] = [128]
    config['value_hidden_layers'] = [128]

    # checkpointing
    config['checkpoint_frequency'] = 200_000

    return config


def generate_model_name(config):
    # generates name for output model
    if OUTPUT_NAME:
        name = OUTPUT_NAME
        return name

    # auto-generate name
    parts = []
    parts.append('ppo')

    # observability
    if config['mask_size'] is None:
        parts.append('full_obs')
    else:
        parts.append(f'partial_obs_m{config["mask_size"]}')

    # reward preset
    parts.append(config['reward_preset_name'])

    # timesteps in millions
    timesteps_m = config['total_timesteps'] / 1_000_000
    parts.append(f'{timesteps_m:.0f}M')

    model_name = '_'.join(parts)
    return model_name


def create_single_environment(config, rank=0):
    # creates one environment
    def _create():
        # create base environment
        # check mask size
        if config['mask_size']:
            actual_mask_size = config['mask_size']
        else:
            actual_mask_size = 0

        env = gym_SnakeEnvironment(
            n_boards=1,
            board_size=config['board_size'],
            mask_size=actual_mask_size,
            rewards=config['rewards'],
        )

        # wrap with monitor for episode tracking
        env = Monitor(env, None)
        return env

    # set seed for this environment
    set_random_seed(config['seed'] + rank)

    return _create


def create_vectorized_environment(config, num_envs=None):
    # creates parallel envs
    if num_envs is None:
        num_envs = config['n_parallel_envs']

    # create environment factory functions
    env_creators = []
    for i in range(num_envs):
        creator = create_single_environment(config, rank=i)
        env_creators.append(creator)

    # use subprocvecenv for multiple envs (parallel), dummyvecenv for single
    if num_envs > 1:
        vec_env = SubprocVecEnv(env_creators)
    else:
        vec_env = DummyVecEnv(env_creators)

    return vec_env


def create_ppo_model(environment, config, device='auto'):
    # creates new ppo model

    # build features extractor kwargs
    features_extractor_kwargs = {}
    features_extractor_kwargs['features_dim'] = config['features_dim']

    # build net arch
    net_arch = {}
    net_arch['pi'] = config['policy_hidden_layers']
    net_arch['vf'] = config['value_hidden_layers']

    # build policy kwargs step by step
    policy_kwargs = {}
    policy_kwargs['features_extractor_class'] = SnakeCNN
    policy_kwargs['features_extractor_kwargs'] = features_extractor_kwargs
    policy_kwargs['net_arch'] = net_arch
    policy_kwargs['normalize_images'] = False

    model = PPO(
        policy='CnnPolicy',
        env=environment,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        clip_range_vf=config['clip_range_vf'],
        normalize_advantage=config['normalize_advantage'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config['seed'],
        device=device,
    )

    return model


def load_existing_model(model_path, environment, device='auto'):
    # loads saved model
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"

    print(f"Loading model from: {model_path}")
    loaded_model = PPO.load(model_path, env=environment, device=device)
    return loaded_model


def train_agent(config, save_directory, model_name, resume_path=None, device='auto'):
    # main training function
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # print training info
    print_training_header(config, save_dir, resume_path, device)

    # create environments
    print("Creating environments...")
    train_env = create_vectorized_environment(config)

    # create or load model
    if resume_path:
        model = load_existing_model(resume_path, train_env, device)
        print(f"Resumed training from: {resume_path}")
    else:
        model = create_ppo_model(train_env, config, device)
        print("Created new model")

    # setup callbacks
    log_frequency = config['n_steps'] * config['n_parallel_envs']
    metrics_callback = MetricsCallback(log_frequency=log_frequency, verbose=1)

    checkpoint_freq = max(config['checkpoint_frequency'] // config['n_parallel_envs'], 1)
    checkpoint_callback = SingleCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=model_name,
        verbose=1,
    )

    callbacks = []
    callbacks.append(metrics_callback)
    callbacks.append(checkpoint_callback)

    # train
    print("\n" + "-" * 60)
    print(" Starting Training...")
    print("-" * 60 + "\n")

    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    training_duration = datetime.now() - start_time

    # save final model
    final_model_path = save_dir / model_name
    model.save(str(final_model_path))
    print(f"\nModel saved to: {final_model_path}.zip")

    # verify model file exists and clean up checkpoints
    model_file_path = str(final_model_path) + ".zip"
    model_exists = os.path.exists(model_file_path)

    if model_exists:
        print("Model file verified, cleaning up checkpoints...")

        # delete checkpoint file if it exists
        checkpoint_path = checkpoint_callback.get_checkpoint_path()
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Deleted checkpoint: {checkpoint_path}")

        # delete checkpoint folder if empty
        if checkpoint_dir.exists():
            try:
                # check if folder is empty
                contents = list(checkpoint_dir.iterdir())
                if len(contents) == 0:
                    checkpoint_dir.rmdir()
                    print(f"Deleted checkpoint folder: {checkpoint_dir}")
                else:
                    print(f"Checkpoint folder not empty, keeping: {checkpoint_dir}")
            except OSError:
                # windows may deny access, not critical
                pass
    else:
        print("Warning: model file not found, keeping checkpoints for recovery")

    # get metrics
    metrics = metrics_callback.get_metrics()

    # save metadata
    metadata = create_training_metadata(
        model_name=model_name,
        config=config,
        metrics=metrics,
        training_duration=training_duration,
        resume_path=resume_path,
    )

    metadata_path = save_dir / f"{model_name}_metadata.json"
    save_json(metadata, metadata_path)
    print(f"Metadata saved to: {metadata_path}")

    # save raw metrics for graph generation
    metrics_path = save_dir / f"{model_name}_metrics.json"
    save_json(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")

    # cleanup
    train_env.close()

    # print summary
    print_training_summary(metrics, training_duration)

    results = {}
    results['model'] = model
    results['model_path'] = str(final_model_path)
    results['metadata'] = metadata
    results['metrics'] = metrics
    return results


def print_training_header(config, save_dir, resume_path, device):
    # prints config info
    print("=" * 60)
    print(" PPO Snake Training")
    print("=" * 60)

    board_size = config['board_size']
    print(f"\nEnvironment:")
    print(f"  Board Size:     {board_size}x{board_size}")

    if config['mask_size']:
        print(f"  Observability:  Partial (mask_size={config['mask_size']})")
    else:
        print(f"  Observability:  Full")

    print(f"  Reward Preset:  {config['reward_preset_name']}")

    print(f"\nTraining:")
    print(f"  Timesteps:      {config['total_timesteps']:,}")
    print(f"  Parallel Envs:  {config['n_parallel_envs']}")
    print(f"  Device:         {device}")

    if resume_path:
        resume_str = resume_path
    else:
        resume_str = 'No'
    print(f"  Resume:         {resume_str}")

    print(f"\nOutput:")
    print(f"  Save Dir:       {save_dir}")
    print()


def print_training_summary(metrics, duration):
    # prints final stats
    print("\n" + "=" * 60)
    print(" Training Complete!")
    print("=" * 60)
    print(f"  Duration:       {duration}")
    print(f"  Total Episodes: {len(metrics['episode_rewards']):,}")

    if metrics['episode_rewards']:
        recent_rewards = metrics['episode_rewards'][-100:]
        recent_lengths = metrics['episode_lengths'][-100:]
        mean_reward = np.mean(recent_rewards)
        mean_length = np.mean(recent_lengths)
        print(f"  Final Mean Reward (last 100): {mean_reward:.2f}")
        print(f"  Final Mean Length (last 100): {mean_length:.1f}")


def create_training_metadata(model_name, config, metrics, training_duration, resume_path):
    # creates metadata dict
    final_stats = {}
    if metrics['episode_rewards']:
        recent_rewards = metrics['episode_rewards'][-100:]
        recent_lengths = metrics['episode_lengths'][-100:]

        final_stats['total_episodes'] = len(metrics['episode_rewards'])
        final_stats['mean_reward_last_100'] = float(np.mean(recent_rewards))
        final_stats['mean_length_last_100'] = float(np.mean(recent_lengths))

    # prepare config for json (handle non-serializable items)
    json_safe_config = {}
    for key, value in config.items():
        if callable(value):
            json_safe_config[key] = str(value)
        else:
            json_safe_config[key] = value

    # build metadata step by step
    metadata = {}
    metadata['model_name'] = model_name
    metadata['timestamp'] = datetime.now().isoformat()
    metadata['training_time_seconds'] = training_duration.total_seconds()
    metadata['config'] = json_safe_config
    metadata['final_stats'] = final_stats
    metadata['resumed_from'] = resume_path

    return metadata


def save_json(data, path):
    # saves dict to json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    # main function

    # build configuration
    config = build_config()

    # generate model name
    model_name = generate_model_name(config)

    # run training
    results = train_agent(
        config=config,
        save_directory=SAVE_DIR,
        model_name=model_name,
        resume_path=RESUME_PATH,
        device=DEVICE,
    )

    print(f"\nTraining complete. Model saved as: {model_name}.zip")


if __name__ == "__main__":
    main()
