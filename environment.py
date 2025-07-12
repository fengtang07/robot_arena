import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco

class WarehouseEnv(gym.Env):
    def __init__(self, max_steps=1000):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path('warehouse.xml')
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        # Track previous distances for reward shaping
        self.prev_robot_to_package_dist = None
        self.prev_package_to_target_dist = None
        self.package_picked_up = False

    def _get_obs(self):
        robot_pos = self.data.qpos[:2]
        robot_vel = self.data.qvel[:2]
        package_pos = self.data.qpos[2:4]  # Package position from its joints
        target_pos = self.data.body('target').xpos[:2]
        shelf1_pos = self.data.body('shelf1').xpos[:2]
        shelf2_pos = self.data.body('shelf2').xpos[:2]
        shelf3_pos = self.data.body('shelf3').xpos[:2]
        return np.concatenate([robot_pos, robot_vel, package_pos, target_pos, shelf1_pos, shelf2_pos, shelf3_pos])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.package_picked_up = False
        
        # Reset robot to center
        self.data.qpos[0] = 0  # robot_x
        self.data.qpos[1] = 0  # robot_y
        
        # Random package position (avoid shelves)
        while True:
            pkg_x = np.random.uniform(-1.5, 1.5)
            pkg_y = np.random.uniform(-1.5, 1.5)
            # Check if position is away from shelves
            if (abs(pkg_x - 0.8) > 0.3 and abs(pkg_x + 0.8) > 0.3 and abs(pkg_y) > 0.3):
                break
        
        self.data.qpos[2] = pkg_x  # package_x
        self.data.qpos[3] = pkg_y  # package_y
        
        # Random target position (away from package and shelves)
        while True:
            tgt_x = np.random.uniform(-1.5, 1.5)
            tgt_y = np.random.uniform(-1.5, 1.5)
            dist_to_package = np.sqrt((tgt_x - pkg_x)**2 + (tgt_y - pkg_y)**2)
            if (dist_to_package > 0.5 and 
                abs(tgt_x - 0.8) > 0.3 and abs(tgt_x + 0.8) > 0.3 and abs(tgt_y) > 0.3):
                break
        
        # Update target position in the model
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        self.model.body_pos[target_body_id][:2] = [tgt_x, tgt_y]
        
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize distance tracking
        robot_pos = self.data.qpos[:2]
        package_pos = self.data.qpos[2:4]
        target_pos = self.data.body('target').xpos[:2]
        self.prev_robot_to_package_dist = np.linalg.norm(robot_pos - package_pos)
        self.prev_package_to_target_dist = np.linalg.norm(package_pos - target_pos)
        
        return self._get_obs(), {}

    def step(self, action):
        # Apply actions
        self.data.ctrl[:2] = action
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get positions
        robot_pos = self.data.qpos[:2]
        package_pos = self.data.qpos[2:4]
        target_pos = self.data.body('target').xpos[:2]
        
        # Calculate distances
        robot_to_package_dist = np.linalg.norm(robot_pos - package_pos)
        package_to_target_dist = np.linalg.norm(package_pos - target_pos)
        
        # Check if package is picked up (robot is close to package)
        if robot_to_package_dist < 0.2:
            self.package_picked_up = True
        
        # If package is picked up, move it with the robot
        if self.package_picked_up:
            # Move package to follow robot with smaller offset to make success easier
            self.data.qpos[2] = robot_pos[0] + 0.05  # Reduced from 0.1 to 0.05
            self.data.qpos[3] = robot_pos[1]         # package_y follows robot exactly
            # Update package position after manual movement
            package_pos = self.data.qpos[2:4]
            package_to_target_dist = np.linalg.norm(package_pos - target_pos)
        
        # Reward calculation with improved shaping to prevent getting stuck
        reward = 0
        
        # 1. Reward for getting closer to package (if not picked up)
        if not self.package_picked_up:
            if robot_to_package_dist < self.prev_robot_to_package_dist:
                reward += 1.0  # Getting closer to package
            else:
                reward -= 0.2  # Moving away from package
        else:
            reward += 2.0  # Bonus for having package
        
        # 2. Strong reward for moving package closer to target (if picked up)
        if self.package_picked_up:
            distance_improvement = self.prev_package_to_target_dist - package_to_target_dist
            if distance_improvement > 0:
                # Scale reward based on improvement - bigger improvements get bigger rewards
                reward += 5.0 * distance_improvement  # Strong incentive for progress
            else:
                reward -= 1.0  # Penalty for moving package away from target
            
            # Additional distance-based reward to encourage getting very close
            if package_to_target_dist < 1.0:
                reward += (1.0 - package_to_target_dist) * 2.0  # Bonus for being close
            
            # Extra incentive when very close to target
            if package_to_target_dist < 0.5:
                reward += (0.5 - package_to_target_dist) * 5.0  # Strong incentive for final approach
        else:
            distance_improvement = 0  # No improvement if package not picked up
        
        # 3. Success reward - increase threshold to make it more achievable
        success = package_to_target_dist < 0.4 and self.package_picked_up
        if success:
            reward += 200  # Massive success bonus
        
        # 4. Reduced step penalty to allow more exploration
        reward -= 0.005  # Smaller penalty
        
        # 5. Penalty for hitting walls
        if abs(robot_pos[0]) > 1.9 or abs(robot_pos[1]) > 1.9:
            reward -= 2.0  # Stronger wall penalty
        
        # 6. Anti-stagnation: penalty for not making progress
        if self.package_picked_up and self.current_step > 100:
            if abs(distance_improvement) < 0.001:  # Not making progress
                reward -= 0.1  # Small penalty for stagnation
        
        # Update tracking
        self.prev_robot_to_package_dist = robot_to_package_dist
        self.prev_package_to_target_dist = package_to_target_dist
        
        terminated = success
        truncated = self.current_step >= self.max_steps
        
        info = {
            'is_success': success,
            'steps': self.current_step,
            'robot_to_package_dist': robot_to_package_dist,
            'package_to_target_dist': package_to_target_dist,
            'package_picked_up': self.package_picked_up
        }
        
        return self._get_obs(), reward, terminated, truncated, info 