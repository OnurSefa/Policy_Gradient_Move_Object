import torch
import torchvision.transforms as transforms
import numpy as np
import environment
from agent import Agent
import os
import mlflow
import torch.multiprocessing as mp
import time

NAME = "0016"
DELTA = 0.05
LEARNING_RATE = 5e-6
GAMMA = 0.99
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
EPSILON = 1
EPSILON_DECAY_RATE = 0.98
EPSILON_DECAY_STEPS = 1
NUM_EPISODES = 10001
DESCRIPTION = "complex state, simple model"

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
    "description": DESCRIPTION
}


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delta = DELTA
        self._goal_thresh = 0.075  # easier goal detection
        self._max_timesteps = 100  # allow more steps
        self._prev_obj_pos = None  # track object movement

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def reset(self):
        super().reset()
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()  # initialize previous position
        self._t = 0

        try:
            return self.high_level_state()
        except:
            return None

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    # def reward(self):
    #
    #     state = self.high_level_state()
    #     ee_pos = state[:2]
    #     obj_pos = state[2:4]
    #     goal_pos = state[4:6]
    #
    #     d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
    #     d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    #
    #     # distance-based rewards
    #     r_ee_to_obj = -0.1 * d_ee_to_obj  # getting closer to object
    #     r_obj_to_goal = -0.2 * d_obj_to_goal  # moving object to goal
    #
    #     # direction bonus
    #     obj_movement = obj_pos - self._prev_obj_pos
    #     dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
    #     r_direction = 0.5 * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
    #     if np.linalg.norm(obj_movement) < 1e-6:  # Avoid division by zero
    #         r_direction = 0.0
    #
    #     # terminal bonus
    #     r_terminal = 10.0 if self.is_terminal() else 0.0
    #
    #     r_step = -0.1  # penalty for each step
    #
    #     self._prev_obj_pos = obj_pos.copy()
    #     return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # Previous object position (for movement calculation)
        obj_movement = obj_pos - self._prev_obj_pos
        obj_movement_magnitude = np.linalg.norm(obj_movement)

        # Shaped rewards with better scaling
        r_ee_to_obj = np.exp(-5.0 * d_ee_to_obj)  # Exponential shaping for contact

        # Progressive reward for object-to-goal distance
        # Higher reward as object gets closer to goal
        r_obj_to_goal = 2.0 * np.exp(-10.0 * d_obj_to_goal)

        # Direction bonus - reward only when object actually moves
        r_direction = 0.0
        if obj_movement_magnitude > 1e-4:
            dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
            r_direction = 3.0 * np.dot(obj_movement / (obj_movement_magnitude + 1e-8), dir_to_goal)

        # Bonus for any object movement (to encourage interaction)
        r_movement = 0.5 * min(obj_movement_magnitude * 10, 1.0)

        # Terminal bonus
        r_terminal = 20.0 if self.is_terminal() else 0.0

        # Small time penalty
        r_step = -0.05

        # Store current object position for next step
        self._prev_obj_pos = obj_pos.copy()

        # Total reward
        total_reward = r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step + r_movement

        return total_reward

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action, numpy_action=False):
        if numpy_action:
            step_action = action * self._delta
        else:
            step_action = action.detach().clamp(-1, 1).cpu().numpy() * self._delta

        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + step_action, [0.25, -0.3], [0.75, 0.3])
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)

        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        if result:  # If the action is successful
            truncated = self.is_truncated()
        else:  # If didn't realize the action
            truncated = True
        return state, reward, terminal, truncated


def train():
    name = NAME

    os.makedirs(f'models_pt/{name}', exist_ok=True)
    os.makedirs(f'metrics_np/{name}', exist_ok=True)

    mlflow.start_run(run_name=f'{name}')
    mlflow.log_params(params=parameters)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    env = Hw3Env(render_mode="offscreen")
    agent = Agent(device, LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY_RATE, EPSILON_DECAY_STEPS, ENTROPY_COEF, MAX_GRAD_NORM)
    mlflow.log_param("model", str(agent.model))
    num_episodes = NUM_EPISODES

    cumulative_rewards = []
    rps = []
    losses = []
    success_rate = []

    success_window = []
    window_size = 100

    for i in range(num_episodes):
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        success = False

        while not done:
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            agent.add_reward(reward)
            done = is_terminal or is_truncated
            if is_terminal and not is_truncated:
                success = True
            cumulative_reward += reward
            state = next_state
            episode_steps += 1

        success_window.append(float(success))
        if len(success_window) > window_size:
            success_window.pop(0)
        current_success_rate = sum(success_window) / len(success_window)
        success_rate.append(current_success_rate)

        print(f"Episode={i}, reward={cumulative_reward}")
        cumulative_rewards.append(cumulative_reward)
        rps.append(cumulative_reward/episode_steps)
        loss = agent.update_model()
        losses.append(loss)

        mlflow.log_metrics({
            "total reward": cumulative_reward,
            "rps": cumulative_reward/episode_steps,
            "loss": loss,
            "epsilon": agent.e,
            "success_rate": current_success_rate,
            "episode_steps": episode_steps,
        }, step=i)

        if i % 250 == 0:
            torch.save(agent.model.state_dict(), f"models_pt/{name}/model_{i:05}.pth")

    np.save(f"metrics_np/{name}/cumulative_rewards.npy", np.array(cumulative_rewards))
    np.save(f"metrics_np/{name}/rps.npy", np.array(rps))
    np.save(f"metrics_np/{name}/losses.npy", np.array(losses))
    np.save(f"metrics_np/{name}/success_rate.npy", np.array(success_rate))


def complex_state(state):
    distances = []
    angles = []
    for i in range(3):
        for j in range(i+1, 3):
            A = state[i*2:i*2+2]
            B = state[j*2:j*2+2]
            distances.append(torch.norm(A - B))

            theta = torch.atan2(B[1] - A[1], B[0] - A[0])
            degrees = torch.rad2deg(theta)
            if 0 <= degrees <= 90:
                x_value = (90 - degrees) / 90
                y_value = degrees / 90
                x_index = 0
                y_index = 1
            elif 90 < degrees <= 180:
                x_value = (degrees - 90) / 90
                y_value = (180 - degrees) / 90
                x_index = 2
                y_index = 1
            elif -90 <= degrees < 0:
                x_value = (90 + degrees) / 90
                y_value = - degrees / 90
                x_index = 0
                y_index = 3
            else:
                x_value = - (90 + degrees) / 90
                y_value = (180 + degrees) / 90
                x_index = 1
                y_index = 3

            angle = torch.zeros(8)
            angle[4+x_index] = 1
            angle[x_index] = x_value
            angle[4+y_index] = 1
            angle[y_index] = y_value

            angles.append(angle)

    distances = torch.stack(distances)
    angles = torch.stack(angles).flatten()
    state = torch.cat([state, distances, angles])
    return state


def collector(model, shared_queue, is_collecting, is_finished, device):
    env = Hw3Env(render_mode="offscreen")
    while not is_finished.is_set():
        while is_collecting.is_set():
            env.reset()
            state = env.high_level_state()
            state = torch.from_numpy(state).float()
            state = complex_state(state)
            done = False
            cum_reward = 0.0

            states = []
            actions = []
            next_states = []
            rewards = []

            while not done:
                states.append(state)
                with torch.no_grad():
                    action_mean, log_std = model(state).chunk(2, dim=-1)
                action_std = torch.exp(log_std) + EPSILON
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.rsample()
                next_state, reward, is_terminal, is_truncated = env.step(action[0])
                next_state = torch.from_numpy(next_state).float()
                next_state = complex_state(next_state)
                cum_reward += reward
                done = is_terminal or is_truncated
                state = next_state
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                if is_finished.is_set():
                    break
            while len(states) < 100:
                states.append(torch.zeros(6))
                actions.append(torch.zeros((1, 2)))
                next_states.append(torch.zeros(6))
                rewards.append(0)

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)

            shared_queue.put((states, actions, next_states, rewards, returns, cum_reward))
            if is_finished.is_set():
                break
        if is_finished.is_set():
            break
        is_collecting.wait()


def train_mp():
    global EPSILON

    name = NAME

    os.makedirs(f'models_pt/{name}', exist_ok=True)
    os.makedirs(f'metrics_np/{name}', exist_ok=True)

    mlflow.start_run(run_name=f'{name}')
    mlflow.log_params(params=parameters)

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device('cpu')
    device = torch.device('cpu')
    agent = Agent(device, LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY_RATE, EPSILON_DECAY_STEPS, ENTROPY_COEF, MAX_GRAD_NORM)
    mlflow.log_param("model", str(agent.model))
    agent.model.share_memory()
    shared_queue = mp.Queue()
    is_collecting = mp.Event()
    is_finished = mp.Event()

    is_collecting.set()
    procs = []
    for i in range(4):
        p = mp.Process(target=collector, args=(agent.model, shared_queue, is_collecting, is_finished, device))
        p.start()
        procs.append(p)

    for i in range(NUM_EPISODES):
        start = time.time()
        buffer_fed = 0
        states_list = []
        actions_list = []
        next_states_list = []
        rewards_list = []
        returns_list = []
        cumulative_rewards = []
        while buffer_fed < 8:
            if not shared_queue.empty():
                states, actions, next_states, rewards, returns, cumulative_reward = shared_queue.get()
                states_list.append(torch.stack(states))
                actions_list.append(torch.cat(actions))
                next_states_list.append(torch.stack(next_states))
                rewards_list.append(torch.FloatTensor(rewards))
                returns_list.append(torch.FloatTensor(returns))
                cumulative_rewards.append(cumulative_reward)
                buffer_fed += 1
        end = time.time()
        is_collecting.clear()
        states = torch.cat(states_list, dim=0)
        actions = torch.cat(actions_list, dim=0)
        # next_states = torch.cat(next_states_list, dim=0)
        # rewards = torch.stack(rewards_list, dim=0)
        returns = torch.stack(returns_list, dim=0)
        print(f"{16/(end-start):.2f} runs/sec... Updating model...")
        loss = agent.update_model_mp(states, actions, returns, EPSILON)

        mlflow.log_metrics({
            "total reward": sum(cumulative_rewards) / len(cumulative_rewards),
            "loss": loss,
            "epsilon": EPSILON
        }, step=i)

        if i % 250 == 0:
            torch.save(agent.model.state_dict(), f"models_pt/{name}/model_{i:05}.pth")

        if i % EPSILON_DECAY_STEPS == 0:
            EPSILON *= EPSILON_DECAY_RATE
        print(i, loss)
        is_collecting.set()
    is_finished.set()
    for p in procs:
        p.join()


if __name__ == "__main__":
    # train()
    train_mp()
