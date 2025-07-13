import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco

class WarehouseEnv(gym.Env):
    def __init__(self, max_steps=1000, reward_config=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path('warehouse.xml')
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.current_step = 0
        
        # The action space is for the robot's motors
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # UPDATED: Observation space now includes the moving obstacle's position (14 -> 16)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        # NEW: Configurable reward system
        if reward_config is None:
            # Default reward values (ENHANCED penalties for better navigation)
            self.reward_config = {
                'success_bonus': 200.0,
                'progress_reward_factor': 5.0,
                'wall_penalty': 5.0,  # Increased from 2.0 to 5.0
                'obstacle_penalty': 8.0,  # Increased from 3.0 to 8.0
                'time_penalty': 0.01  # Increased from 0.005 to 0.01
            }
        else:
            self.reward_config = reward_config

        # Track previous distances for reward shaping
        self.prev_robot_to_package_dist = None
        self.prev_package_to_target_dist = None
        self.package_picked_up = False
        
        # NEW: Track robot movement for stagnation detection
        self.robot_position_history = []
        self.stagnation_threshold = 0.05  # Minimum movement per step
        self.max_stagnation_steps = 50   # Max steps without progress

    def _get_obs(self):
        robot_pos = self.data.qpos[:2]
        robot_vel = self.data.qvel[:2]
        package_pos = self.data.qpos[2:4]
        target_pos = self.data.body('target').xpos[:2]
        shelf1_pos = self.data.body('shelf1').xpos[:2]
        shelf2_pos = self.data.body('shelf2').xpos[:2]
        shelf3_pos = self.data.body('shelf3').xpos[:2]
        # NEW: Get the moving obstacle's position
        moving_obstacle_pos = self.data.body('moving_obstacle').xpos[:2]
        
        # UPDATED: Concatenate all observations
        return np.concatenate([
            robot_pos, robot_vel, package_pos, target_pos, 
            shelf1_pos, shelf2_pos, shelf3_pos, moving_obstacle_pos
        ])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.package_picked_up = False
        
        # Reset robot to center
        self.data.qpos[0] = 0
        self.data.qpos[1] = 0
        
        # RANDOM starting position for the package (avoid obstacles)
        while True:
            pkg_x = np.random.uniform(-3.5, 3.5)
            pkg_y = np.random.uniform(-3.5, 3.5)
            # Avoid shelves and southern obstacle area
            if (abs(pkg_x - 1.5) > 0.3 and abs(pkg_x + 1.5) > 0.3 and  # avoid shelf1 & shelf2
                abs(pkg_y - 1.5) > 0.3 and abs(pkg_y + 2.0) > 0.5):     # avoid shelf3 & moving obstacle
                break
        
        self.data.qpos[2] = pkg_x
        self.data.qpos[3] = pkg_y
        
        # RANDOM target position (avoid obstacles and ensure distance from package)
        while True:
            tgt_x = np.random.uniform(-3.5, 3.5)
            tgt_y = np.random.uniform(-3.5, 3.5)
            # Avoid shelves, southern obstacle area, and ensure minimum distance from package
            distance_from_package = np.sqrt((tgt_x - pkg_x)**2 + (tgt_y - pkg_y)**2)
            if (abs(tgt_x - 1.5) > 0.3 and abs(tgt_x + 1.5) > 0.3 and  # avoid shelf1 & shelf2
                abs(tgt_y - 1.5) > 0.3 and abs(tgt_y + 2.0) > 0.5 and  # avoid shelf3 & moving obstacle
                distance_from_package > 2.0):                           # minimum distance from package
                break
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        self.model.body_pos[target_body_id][:2] = [tgt_x, tgt_y]
        
        # Give the moving obstacle an initial velocity to eliminate startup delay
        obstacle_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'obstacle_slide')
        if obstacle_joint_id >= 0:
            # Set initial velocity based on the first control signal
            initial_control = np.sin(0 * 5.0 + np.pi/2) * 0.95  # Same as first step
            self.data.qvel[obstacle_joint_id] = initial_control * 2.0  # Initial velocity
        
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize distance tracking
        robot_pos = self.data.qpos[:2]
        package_pos = self.data.qpos[2:4]
        target_pos = self.data.body('target').xpos[:2]
        self.prev_robot_to_package_dist = np.linalg.norm(robot_pos - package_pos)
        self.prev_package_to_target_dist = np.linalg.norm(package_pos - target_pos)
        
        # NEW: Initialize movement tracking
        self.robot_position_history = [robot_pos.copy()]
        
        return self._get_obs(), {}

    def step(self, action):
        # NEW: Control the moving obstacle with a simple sine wave for back-and-forth motion
        # The third actuator (index 2) is the obstacle_motor
        # PERFECTLY CONTROLLED movement - COMPENSATE FOR 2-FRAME PHYSICS DELAY
        # Start sine wave 2 steps ahead to account for MuJoCo integration delay
        sine_value = np.sin((self.current_step + 2) * 1.0 + np.pi/2) * 0.95  # Well-controlled speed (/5 from original)
        self.data.ctrl[2] = sine_value

        # Apply actions to the robot's motors (index 0 and 1)
        self.data.ctrl[:2] = action
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get positions
        robot_pos = self.data.qpos[:2]
        package_pos = self.data.qpos[2:4]
        target_pos = self.data.body('target').xpos[:2]
        moving_obstacle_pos = self.data.body('moving_obstacle').xpos[:2]
        
        # NEW: Track robot movement for stagnation detection
        self.robot_position_history.append(robot_pos.copy())
        if len(self.robot_position_history) > self.max_stagnation_steps:
            self.robot_position_history.pop(0)  # Keep only recent positions
        
        # Calculate distances
        robot_to_package_dist = np.linalg.norm(robot_pos - package_pos)
        package_to_target_dist = np.linalg.norm(package_pos - target_pos)
        robot_to_obstacle_dist = np.linalg.norm(robot_pos - moving_obstacle_pos)
        
        # Check if package is picked up
        if robot_to_package_dist < 0.2:
            self.package_picked_up = True
        
        # If package is picked up, move it with the robot
        if self.package_picked_up:
            self.data.qpos[2] = robot_pos[0] + 0.05
            self.data.qpos[3] = robot_pos[1]
            package_pos = self.data.qpos[2:4]
            package_to_target_dist = np.linalg.norm(package_pos - target_pos)
        
        # --- UPDATED: Reward calculation using the configurable values ---
        reward = 0
        
        # 1. Reward for getting closer to package (if not picked up)
        if not self.package_picked_up:
            if robot_to_package_dist < self.prev_robot_to_package_dist:
                reward += 1.0
            else:
                reward -= 0.2
        else:
            reward += 2.0  # Bonus for having package
        
        # 2. Reward for moving package closer to target (if picked up)
        if self.package_picked_up:
            distance_improvement = self.prev_package_to_target_dist - package_to_target_dist
            if distance_improvement > 0:
                reward += self.reward_config['progress_reward_factor'] * distance_improvement
            else:
                reward -= 1.0
            
            # Additional distance-based reward to encourage getting very close
            if package_to_target_dist < 1.0:
                reward += (1.0 - package_to_target_dist) * 2.0
            
            # Extra incentive when very close to target
            if package_to_target_dist < 0.5:
                reward += (0.5 - package_to_target_dist) * 5.0
        
        # 3. Success reward
        success = package_to_target_dist < 0.4 and self.package_picked_up
        if success:
            reward += self.reward_config['success_bonus']
        
        # 4. Time penalty
        reward -= self.reward_config['time_penalty']
        
        # 5. ENHANCED: Stronger penalty for hitting walls
        if abs(robot_pos[0]) > 3.8 or abs(robot_pos[1]) > 3.8:
            reward -= self.reward_config['wall_penalty'] * 3  # Much stronger penalty
            # Additional penalty for being very close to walls
            if abs(robot_pos[0]) > 3.9 or abs(robot_pos[1]) > 3.9:
                reward -= self.reward_config['wall_penalty'] * 5  # Even stronger

        # 6. NEW: Penalty for getting too close to the moving obstacle
        if robot_to_obstacle_dist < 0.3:  # 0.1 (robot) + 0.2 (obstacle)
            reward -= self.reward_config['obstacle_penalty']
        
        # 7. ENHANCED: Anti-stagnation detection (works with or without package)
        if len(self.robot_position_history) >= 10:  # Need at least 10 positions to check
            # Check if robot has moved significantly in the last 10 steps
            recent_positions = self.robot_position_history[-10:]
            max_distance_moved = 0
            for i in range(1, len(recent_positions)):
                dist = np.linalg.norm(recent_positions[i] - recent_positions[i-1])
                max_distance_moved = max(max_distance_moved, dist)
            
            # If robot barely moved in 10 steps, it's likely stuck
            if max_distance_moved < self.stagnation_threshold:
                reward -= 5.0  # Strong penalty for being stuck
                
                # Even stronger penalty if stuck for longer
                if len(self.robot_position_history) >= 20:
                    recent_20_positions = self.robot_position_history[-20:]
                    total_distance_20 = 0
                    for i in range(1, len(recent_20_positions)):
                        total_distance_20 += np.linalg.norm(recent_20_positions[i] - recent_20_positions[i-1])
                    
                    if total_distance_20 < 0.5:  # Very little movement in 20 steps
                        reward -= 10.0  # Very strong penalty for prolonged stagnation
        
        # 8. Additional penalty for package delivery stagnation
        if self.package_picked_up and self.current_step > 100:
            distance_improvement = self.prev_package_to_target_dist - package_to_target_dist
            if abs(distance_improvement) < 0.001:  # Not making progress
                reward -= 2.0  # Increased penalty for delivery stagnation
        
        # Update tracking
        self.prev_robot_to_package_dist = robot_to_package_dist
        self.prev_package_to_target_dist = package_to_target_dist
        
        # Enhanced termination conditions
        terminated = success
        
        # NEW: Early termination for severely stuck robots
        severely_stuck = False
        if len(self.robot_position_history) >= self.max_stagnation_steps:
            # Check if robot has been stuck for the max allowed steps
            total_movement = 0
            for i in range(1, len(self.robot_position_history)):
                total_movement += np.linalg.norm(self.robot_position_history[i] - self.robot_position_history[i-1])
            
            # If robot moved less than 1 unit in max_stagnation_steps, terminate
            if total_movement < 1.0:
                severely_stuck = True
                terminated = True
        
        truncated = self.current_step >= self.max_steps
        
        info = {
            'is_success': success,
            'steps': self.current_step,
            'robot_to_package_dist': robot_to_package_dist,
            'package_to_target_dist': package_to_target_dist,
            'package_picked_up': self.package_picked_up,
            'robot_to_obstacle_dist': robot_to_obstacle_dist,
            'severely_stuck': severely_stuck,  # NEW: Track if terminated due to being stuck
            'final_obs': self._get_obs()  # So we can visualize it later
        }
        
        return self._get_obs(), reward, terminated, truncated, info 