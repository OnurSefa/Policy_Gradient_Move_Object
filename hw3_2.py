import torch
import numpy as np
import os
import mlflow
from tqdm import tqdm
from agent import SACAgent
from homework3 import Hw3Env

NAME = "0000"
DELTA = 0.05
LEARNING_RATE = 3e-4
GAMMA = 0.99
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
EPSILON = 1
EPSILON_DECAY_RATE = 0.98
EPSILON_DECAY_STEPS = 1
NUM_EPISODES = 10001
DESCRIPTION = "Soft Actor Critic (SAC) Implementation"
TAU = 0.005
ALPHA = 0.2
BATCH_SIZE = 256
BUFFER_SIZE = 100000
UPDATES_PER_STEP = 1
START_UPDATES = 1000
RANDOM_STEPS = 1000
TARGET_UPDATE_INTERVAL = 1
AUTOMATIC_ENTROPY_TUNING = True

parameters = {
    "name": NAME,
    "delta": DELTA,
    "learning_rate": LEARNING_RATE,
    "gamma": GAMMA,
    'entropy_coef': ENTROPY_COEF,
    'max_grad_norm': MAX_GRAD_NORM,
    "epsilon": EPSILON,
    "epsilon_decay_rate": EPSILON_DECAY_RATE,
    "epsilon_decay_steps": EPSILON_DECAY_STEPS,
    "num_episodes": NUM_EPISODES,
    "description": DESCRIPTION,
    "tau": TAU,
    "alpha": ALPHA,
    "batch_size": BATCH_SIZE,
    "buffer_size": BUFFER_SIZE,
    "updates_per_step": UPDATES_PER_STEP,
    "start_updates": START_UPDATES,
    "target_update_interval": TARGET_UPDATE_INTERVAL,
    "automatic_entropy_tuning": AUTOMATIC_ENTROPY_TUNING,
}


def train():
    name = "SAC_" + NAME

    # Create directories
    os.makedirs(f'models_sac_pt/{name}', exist_ok=True)
    os.makedirs(f'metrics_sac_np/{name}', exist_ok=True)

    mlflow.start_run(run_name=f'{name}')
    mlflow.log_params(parameters)

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    env = Hw3Env(render_mode="offscreen")

    # Create SAC agent
    agent = SACAgent(
        state_dim=6,
        action_dim=2,
        hidden_dim=256,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        automatic_entropy_tuning=AUTOMATIC_ENTROPY_TUNING,
        epsilon=EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        device=device
    )

    mlflow.log_param("policy_model", str(agent.policy))
    mlflow.log_param("critic_model", str(agent.critic))

    total_rewards = []
    episode_lengths = []
    success_history = []
    avg_critic_losses = []
    avg_policy_losses = []
    avg_alpha_losses = []
    success_window = []

    total_steps = 0
    for episode in range(NUM_EPISODES):
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        success = False

        while not done:
            if np.random.random() < agent.epsilon:
                action = np.random.uniform(-1, 1, size=2)
            else:
                action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action, True)
            done = is_terminal or is_truncated
            if is_terminal and not is_truncated:
                success = True

            agent.add_experience(state, action, reward, next_state, done)
            if total_steps >= START_UPDATES:
                critic_loss, policy_loss, alpha_loss = agent.update_model(UPDATES_PER_STEP)
                avg_critic_losses.append(critic_loss)
                avg_policy_losses.append(policy_loss)
                avg_alpha_losses.append(alpha_loss)

            state = next_state
            cumulative_reward += reward
            episode_steps += 1
            total_steps += 1

        success_window.append(float(success))
        if len(success_window) > 100:
            success_window.pop(0)
        current_success_rate = sum(success_window) / len(success_window)
        success_history.append(current_success_rate)
        total_rewards.append(cumulative_reward)
        episode_lengths.append(episode_steps)

        mlflow.log_metrics({
            "cumulative_reward": cumulative_reward,
            "episode_length": episode_steps,
            "success": int(success),
            "success_rate": current_success_rate,
            "total_steps": total_steps,
            "critic_loss": np.mean(avg_critic_losses[-100:]) if avg_critic_losses else 0,
            "policy_loss": np.mean(avg_policy_losses[-100:]) if avg_policy_losses else 0,
            "alpha_loss": np.mean(avg_alpha_losses[-100:]) if avg_alpha_losses else 0,
            "alpha": agent.log_alpha.exp().item() if agent.automatic_entropy_tuning else agent.alpha
        }, step=episode)

        if episode % 500 == 0:
            agent.save(f"models_sac_pt/{name}/model_{episode:05}.pth")
            np.save(f"metrics_sac_np/{name}/rewards.npy", np.array(total_rewards))
            np.save(f"metrics_sac_np/{name}/lengths.npy", np.array(episode_lengths))
            np.save(f"metrics_sac_np/{name}/success_rate.npy", np.array(success_history))


if __name__ == '__main__':
    train()

