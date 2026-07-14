# training menu script

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_TIMESTEPS = 6_000_000

# remember that train_ppo contains all the variables to optimize for training rig 


def get_timestamp():
    # returns current time as string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


def print_header(message):
    # prints a header
    print("\n" + "=" * 70)
    print(f" {message}")
    print("=" * 70)


def print_status(message):
    # prints status message
    print(f"[{get_timestamp()}] {message}")


def format_duration(seconds):
    # formats seconds as readable time
    duration = str(timedelta(seconds=int(seconds)))
    return duration


def run_training(output_name, mask_size, reward_preset, timesteps):
    # runs training with specified configuration
    import source.train_ppo as train_ppo

    print_status("Configuring training...")

    # set configuration variables
    train_ppo.OUTPUT_NAME = output_name
    train_ppo.TIMESTEPS = timesteps
    train_ppo.MASK_SIZE = mask_size
    train_ppo.REWARD_PRESET = reward_preset

    print_status("Starting training...")
    start_time = time.time()

    # build configuration
    config = train_ppo.build_config()

    # generate model name
    model_name = train_ppo.generate_model_name(config)

    # run training
    results = train_ppo.train_agent(
        config=config,
        save_directory=train_ppo.SAVE_DIR,
        model_name=model_name,
        resume_path=train_ppo.RESUME_PATH,
        device=train_ppo.DEVICE,
    )

    elapsed = time.time() - start_time
    print_status(f"Training completed in {format_duration(elapsed)}")
    return True


def run_single_training(config_name, timesteps):
    # trains one model based on config name
    print_header(f"Training: {config_name}")

    if config_name == "ppo_partial_obs_default_rewards":
        return run_training(
            output_name=config_name,
            mask_size=2,
            reward_preset='default_rewards',
            timesteps=timesteps,
        )
    elif config_name == "ppo_full_obs_default_rewards":
        return run_training(
            output_name=config_name,
            mask_size=None,
            reward_preset='default_rewards',
            timesteps=timesteps,
        )
    elif config_name == "ppo_full_obs_step_penalty":
        return run_training(
            output_name=config_name,
            mask_size=None,
            reward_preset='step_penalty',
            timesteps=timesteps,
        )
    elif config_name == "ppo_full_obs_high_penalty":
        return run_training(
            output_name=config_name,
            mask_size=None,
            reward_preset='high_penalty',
            timesteps=timesteps,
        )
    else:
        print_status(f"Unknown config: {config_name}")
        return False


def run_multiple_training(config_names, timesteps):
    # trains multiple models
    results = []
    total_start = time.time()

    print_header(f"TRAINING PIPELINE - {len(config_names)} models @ {timesteps:,} timesteps each")
    print_status(f"Estimated total time: {len(config_names) * 45}-{len(config_names) * 60} minutes")
    print_status("Models will be saved to: weights/")
    print_status("Existing models will be OVERWRITTEN")

    for i, config_name in enumerate(config_names, 1):
        print_header(f"Training {i}/{len(config_names)}: {config_name}")

        success = run_single_training(config_name, timesteps)
        results.append((config_name, success))

        if not success:
            print_status(f"Training failed for {config_name}. Continuing with next model...")

    total_elapsed = time.time() - total_start
    print_header("TRAINING SUMMARY")
    print_status(f"Total training time: {format_duration(total_elapsed)}")

    for name, success in results:
        if success:
            status = "SUCCESS"
        else:
            status = "FAILED"
        print(f"  - {name}: {status}")

    return results


def display_menu():
    # shows menu and gets choice
    print("\n" + "=" * 70)
    print(" MENU")
    print("=" * 70)
    print()
    print("  1. Run training of best PPO model")
    print("     (Partial observability with default rewards)")
    print()
    print("  2. Run training of ALL models for report")
    print("     (Partial obs, full obs, step penalty, high penalty)")
    print()
    print("  3. Exit")
    print()
    print("=" * 70)

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def get_timesteps_input():
    # gets timesteps from user
    print(f"\nDefault timesteps: {DEFAULT_TIMESTEPS:,}")

    while True:
        user_input = input(f"Enter timesteps: ").strip()

        if user_input == '':
            return DEFAULT_TIMESTEPS

        try:
            timesteps = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

        if timesteps <= 0:
            print("Timesteps must be a positive integer.")
            continue

        return timesteps


def confirm_action(message):
    # asks user yes/no
    while True:
        response = input(f"\n{message} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def main():
    # main function
    # verify we're in the right directory
    if not Path("source/train_ppo.py").exists():
        print("ERROR: source/train_ppo.py not found. Run this script from the project root.")
        sys.exit(1)

    while True:
        choice = display_menu()

        if choice == 1:
            # train best model (partial obs with default rewards)
            print_header("BEST MODEL TRAINING")
            print("Model: ppo_partial_obs_default_rewards")
            print("Config: Partial observability (mask_size=2), default rewards")

            timesteps = get_timesteps_input()

            if not confirm_action(f"Start training with {timesteps:,} timesteps?"):
                continue

            success = run_single_training("ppo_partial_obs_default_rewards", timesteps)

            if success:
                print_header("TRAINING COMPLETE")
                print("Model saved to: weights/ppo_partial_obs_default_rewards.zip")

        elif choice == 2:
            # train all models for report
            all_configs = []
            all_configs.append("ppo_partial_obs_default_rewards")
            all_configs.append("ppo_full_obs_default_rewards")
            all_configs.append("ppo_full_obs_step_penalty")
            all_configs.append("ppo_full_obs_high_penalty")

            print_header("ALL MODELS TRAINING")
            print("Models to train:")
            for name in all_configs:
                print(f"  - {name}")

            timesteps = get_timesteps_input()

            if not confirm_action(f"Start training {len(all_configs)} models with {timesteps:,} timesteps each?"):
                continue

            results = run_multiple_training(all_configs, timesteps)

            successes = 0
            for name, success in results:
                if success:
                    successes += 1

            print_header("TRAINING COMPLETE")
            print(f"Successfully trained: {successes}/{len(results)} models")

        elif choice == 3:
            print("\nExiting...")
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
