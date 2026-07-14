# snake rl evaluation script

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from source.gym_environments import gym_SnakeEnvironment
from source.baseline_agent import BaselineAgent
from source.export_graphs import export_all_graphs



EXPORT_FOR_FIGURES = False
N_EPISODES = 100


# configuration
EVALUATE_BASELINE = True
BOARD_SIZE = 10
MAX_STEPS = 1000
OUTPUT_JSON = None
VERBOSE = False
SEED = 0
WIN_REWARD = 1.0
FRUIT_REWARD = 0.5


# helper functions

def discover_models(weights_dir='weights'):
    # find all .zip model files in the weights directory
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"weights directory '{weights_dir}' not found")
        empty_list = []
        return empty_list

    model_files = sorted(weights_path.glob('*.zip'))
    model_paths = []
    for model_file in model_files:
        model_path = str(model_file)
        model_paths.append(model_path)
    return model_paths


MODEL_PATHS = discover_models('weights')


def evaluate_agent(agent, environment, num_episodes, max_steps, verbose=True, agent_name="Agent"):
    # runs evaluation episodes and returns stats
    episode_rewards = []
    episode_lengths = []
    fruits_collected = []
    wins = 0

    for episode in range(num_episodes):
        observation, info = environment.reset(seed=SEED + episode)

        episode_reward = 0.0
        episode_fruits = 0

        for step in range(max_steps):
            # get action from agent
            action = get_agent_action(agent, observation)

            # take step
            observation, reward, terminated, truncated, info = environment.step(action)

            episode_reward += reward

            # count fruits (positive reward above threshold)
            if reward >= FRUIT_REWARD - 0.01:
                episode_fruits += 1

            # check for win (high positive reward)
            if reward >= WIN_REWARD - 0.01:
                wins += 1

            if terminated or truncated:
                break

        # record episode results
        episode_length = step + 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        fruits_collected.append(episode_fruits)

        # print progress
        if verbose and (episode + 1) % 20 == 0:
            print(f"    [{agent_name}] Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Fruits={episode_fruits}, Steps={episode_length}")

    # calculate statistics
    results = calculate_statistics(
        agent_name=agent_name,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        fruits_collected=fruits_collected,
        wins=wins,
        num_episodes=num_episodes,
    )

    return results


def get_agent_action(agent, observation):
    # gets action from agent
    # check if agent supports deterministic parameter (ppo does, baseline doesn't)
    if hasattr(agent, 'policy'):
        # ppo model - use deterministic
        action, _ = agent.predict(observation, deterministic=True)
    else:
        # baseline agent - doesn't support deterministic
        action, _ = agent.predict(observation)

    # convert to integer if needed
    if isinstance(action, np.ndarray):
        action = int(action.item())
    else:
        action = int(action)

    return action


def calculate_statistics(agent_name, episode_rewards, episode_lengths, fruits_collected, wins, num_episodes):
    # calculates mean, std, etc from episode data
    return {
        'agent_name': agent_name,
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'mean_fruits': float(np.mean(fruits_collected)),
        'std_fruits': float(np.std(fruits_collected)),
        'win_rate': float(wins / num_episodes),
        'wins': wins,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'fruits_collected': fruits_collected,
    }


def load_ppo_model(model_path):
    # loads ppo model from file
    model = PPO.load(model_path)
    return model


def create_baseline_agent():
    # creates baseline agent
    agent = BaselineAgent(verbose=0)
    return agent


def create_environment(board_size, mask_size=None):
    # creates gym environment
    # check if mask size was provided
    if mask_size:
        actual_mask_size = mask_size
    else:
        actual_mask_size = 0

    env = gym_SnakeEnvironment(
        n_boards=1,
        board_size=board_size,
        mask_size=actual_mask_size,
    )
    return env


def detect_model_mask_size(model_path):
    # tries to get mask size from metadata
    metadata_path = model_path.replace('.zip', '_metadata.json')
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            config = metadata.get('config', {})
            mask_size = config.get('mask_size')
            if mask_size is not None:
                return mask_size

    # fallback: infer from model name
    if 'partial_obs' in model_path:
        return 2
    return None


def print_results(results):
    # prints results for one agent
    print(f"\n  {results['agent_name']} Results:")
    print(f"  {'-' * 45}")
    print(f"  Mean Reward:    {results['mean_reward']:>8.2f} +/- {results['std_reward']:.2f}")
    print(f"  Mean Length:    {results['mean_length']:>8.1f} +/- {results['std_length']:.1f}")
    print(f"  Mean Fruits:    {results['mean_fruits']:>8.2f} +/- {results['std_fruits']:.2f}")
    print(f"  Win Rate:       {results['win_rate']:>8.2%} ({results['wins']}/{results['num_episodes']})")


def print_comparison(all_results):
    # prints comparison table
    print("\n" + "=" * 60)
    print(" COMPARISON")
    print("=" * 60)

    # header
    header = f"{'Agent':<25} {'Mean Reward':>12} {'Mean Fruits':>12} {'Win Rate':>12}"
    print(f"\n  {header}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")

    # sort by mean reward
    sorted_results = []
    for result in all_results:
        sorted_results.append(result)
    # sort the list by mean reward (highest first)
    for i in range(len(sorted_results)):
        for j in range(i + 1, len(sorted_results)):
            if sorted_results[j]['mean_reward'] > sorted_results[i]['mean_reward']:
                temp = sorted_results[i]
                sorted_results[i] = sorted_results[j]
                sorted_results[j] = temp

    # print each agent
    for result in sorted_results:
        name = result['agent_name'][:25]
        reward = f"{result['mean_reward']:.2f}"
        fruits = f"{result['mean_fruits']:.2f}"
        win_rate = f"{result['win_rate']:.2%}"
        print(f"  {name:<25} {reward:>12} {fruits:>12} {win_rate:>12}")

    # print improvement over baseline if baseline exists
    # find baseline result in all_results
    baseline_result = None
    for r in all_results:
        if r['agent_name'] == 'Baseline':
            baseline_result = r
            break

    if baseline_result:
        print(f"\n  Improvement over Baseline:")
        for result in sorted_results:
            if result['agent_name'] != 'Baseline':
                reward_diff = result['mean_reward'] - baseline_result['mean_reward']
                if baseline_result['mean_reward'] != 0:
                    reward_pct = (reward_diff / abs(baseline_result['mean_reward'])) * 100
                else:
                    reward_pct = 0
                print(f"    {result['agent_name']}: {reward_diff:+.2f} ({reward_pct:+.1f}%)")


def save_results_json(all_results, config, output_path, include_episode_data=False):
    # saves results to json file
    output = {
        'evaluation_date': datetime.now().isoformat(),
        'config': {
            'board_size': config['board_size'],
            'n_episodes': config['n_episodes'],
            'max_steps': config['max_steps'],
            'seed': SEED,
        },
        'results': {},
        'comparisons': {},
    }

    # add each agent's results
    for result in all_results:
        agent_key = result['agent_name'].lower().replace(' ', '_')
        agent_data = {
            'mean_reward': result['mean_reward'],
            'std_reward': result['std_reward'],
            'mean_fruits': result['mean_fruits'],
            'std_fruits': result['std_fruits'],
            'mean_length': result['mean_length'],
            'win_rate': result['win_rate'],
            'wins': result['wins'],
        }

        # include raw episode data for figure generation if enabled
        if include_episode_data:
            agent_data['episode_rewards'] = result['episode_rewards']

        output['results'][agent_key] = agent_data

    # calculate comparisons if baseline exists
    # find baseline result in all_results
    baseline_result = None
    for r in all_results:
        if r['agent_name'] == 'Baseline':
            baseline_result = r
            break

    if baseline_result:
        for result in all_results:
            if result['agent_name'] != 'Baseline':
                agent_key = result['agent_name'].lower().replace(' ', '_')
                if baseline_result['mean_reward'] != 0:
                    improvement = ((result['mean_reward'] - baseline_result['mean_reward'])
                                   / abs(baseline_result['mean_reward'])) * 100
                else:
                    improvement = 0
                output['comparisons'][f'{agent_key}_vs_baseline_pct'] = round(improvement, 2)

    # save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def run_evaluation(config):
    # runs full evaluation loop
    all_results = []

    # evaluate baseline (always uses full observability)
    if config['evaluate_baseline']:
        print("\n" + "-" * 60)
        print(" Evaluating Baseline Agent...")
        print("-" * 60)

        # create full-obs environment for baseline
        env = create_environment(
            board_size=config['board_size'],
            mask_size=None,
        )

        baseline_agent = create_baseline_agent()
        baseline_results = evaluate_agent(
            agent=baseline_agent,
            environment=env,
            num_episodes=config['n_episodes'],
            max_steps=config['max_steps'],
            verbose=config['verbose'],
            agent_name="Baseline",
        )
        all_results.append(baseline_results)
        print_results(baseline_results)
        env.close()

    # evaluate each model
    for model_path in config['model_paths']:
        print("\n" + "-" * 60)
        print(f" Evaluating: {model_path}")
        print("-" * 60)

        # check if model exists
        if not Path(model_path).exists():
            print(f"    Model not found: {model_path}")
            continue

        # detect mask size from model metadata
        model_mask_size = detect_model_mask_size(model_path)
        if model_mask_size:
            print(f"    Detected partial observability: mask_size={model_mask_size}")

        # create environment matching model's observation space
        env = create_environment(
            board_size=config['board_size'],
            mask_size=model_mask_size,
        )

        # load model
        model = load_ppo_model(model_path)
        if model is None:
            env.close()
            continue

        # generate agent name from filename
        agent_name = Path(model_path).stem
        if agent_name.endswith('.zip'):
            agent_name = agent_name[:-4]

        # evaluate
        results = evaluate_agent(
            agent=model,
            environment=env,
            num_episodes=config['n_episodes'],
            max_steps=config['max_steps'],
            verbose=config['verbose'],
            agent_name=agent_name,
        )
        all_results.append(results)
        print_results(results)
        env.close()

    return all_results


def main():
    # main function

    # build configuration
    config = {}
    config['model_paths'] = MODEL_PATHS
    config['evaluate_baseline'] = EVALUATE_BASELINE
    config['n_episodes'] = N_EPISODES
    config['board_size'] = BOARD_SIZE
    config['max_steps'] = MAX_STEPS
    config['verbose'] = VERBOSE

    # print header
    print("\n" + "=" * 60)
    print(" Snake RL - Agent Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Board Size:  {config['board_size']}x{config['board_size']}")
    print(f"  Episodes:    {config['n_episodes']}")
    print(f"  Max Steps:   {config['max_steps']}")
    print(f"  Models:      {len(config['model_paths'])}")

    if EXPORT_FOR_FIGURES:
        export_str = 'Enabled'
    else:
        export_str = 'Disabled'
    print(f"  Export Mode: {export_str}")

    # run evaluation
    all_results = run_evaluation(config)

    # print comparison if multiple agents
    if len(all_results) > 1:
        print_comparison(all_results)

    print("\n" + "=" * 60)

    if EXPORT_FOR_FIGURES:
        print("\n" + "-" * 60)
        print(" Exporting data for figure generation...")
        print("-" * 60)

        # determine output path
        if OUTPUT_JSON:
            output_path = OUTPUT_JSON
        else:
            output_path = "figures/metrics/evaluation_results.json"

        # save with episode data included for reward distribution plots
        save_results_json(
            all_results,
            config,
            output_path,
            include_episode_data=True
        )

        # automatically generate all graphs
        export_all_graphs(output_dir="figures", weights_dir="weights")

    elif OUTPUT_JSON:
        # manual export requested but export_for_figures is false
        # save without episode data (smaller file)
        save_results_json(
            all_results,
            config,
            OUTPUT_JSON,
            include_episode_data=False
        )

    print()


if __name__ == "__main__":
    main()
