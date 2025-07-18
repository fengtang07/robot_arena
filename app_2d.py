import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO
from environment import WarehouseEnv
from training import train_user_robot
import json

st.set_page_config(
    page_title="Warehouse Robot Competition (2D)",
    page_icon="🤖",
    layout="wide"
)

# --- NEW: Leaderboard Functions ---
LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    """Loads the leaderboard from a JSON file."""
    if not os.path.exists(LEADERBOARD_FILE):
        return {}
    try:
        with open(LEADERBOARD_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def update_leaderboard(username, steps):
    """Updates the leaderboard with a new score, keeping only the best."""
    leaderboard = load_leaderboard()
    # Update score only if it's better than the existing one
    if username not in leaderboard or steps < leaderboard[username]:
        leaderboard[username] = steps
        # Sort the leaderboard by steps (ascending)
        sorted_leaderboard = dict(sorted(leaderboard.items(), key=lambda item: item[1]))
        with open(LEADERBOARD_FILE, 'w') as f:
            json.dump(sorted_leaderboard, f, indent=4)
        return True
    return False

def clip_position(pos, min_val=-4.0, max_val=4.0):
    """Clip position data to prevent out-of-bounds visualization issues"""
    return [max(min_val, min(max_val, pos[0])), max(min_val, min(max_val, pos[1]))]

def create_dual_robot_animation_2d(user_path, default_path, user_name="User Robot", default_name="Default Robot", show_full_path=True):
    """Create a 2D animated visualization showing both robots competing"""
    if not user_path or not default_path or not user_path['robot_pos'] or not default_path['robot_pos']:
        return go.Figure()
    
    if show_full_path:
        # Show all steps to see complete journey including getting stuck
        min_length = min(len(user_path['robot_pos']), len(default_path['robot_pos']))
        
        # Use all available data, but sample if too long for smooth animation
        if min_length > 300:
            step = min_length // 300  # Sample to get ~300 frames max
            user_robot_pos = user_path['robot_pos'][::step]
            user_package_pos = user_path['package_pos'][::step]
            default_robot_pos = default_path['robot_pos'][::step]
            default_package_pos = default_path['package_pos'][::step]
        else:
            # Use all data if reasonable length
            user_robot_pos = user_path['robot_pos'][:min_length]
            user_package_pos = user_path['package_pos'][:min_length]
            default_robot_pos = default_path['robot_pos'][:min_length]
            default_package_pos = default_path['package_pos'][:min_length]
    else:
        # Original meaningful motion detection logic
        def find_meaningful_length(robot_positions, threshold=0.01):
            """Find the last frame where the robot was significantly moving"""
            if len(robot_positions) < 2:
                return len(robot_positions)
            
            last_movement = 0
            for i in range(1, len(robot_positions)):
                prev_pos = np.array(robot_positions[i-1][:2])  # x, y only
                curr_pos = np.array(robot_positions[i][:2])
                
                # Calculate movement distance
                movement = np.linalg.norm(curr_pos - prev_pos)
                
                if movement > threshold:
                    last_movement = i
            
            # Add a buffer of 10 frames after last movement to show final position
            return min(len(robot_positions), last_movement + 10)
        
        # Find meaningful lengths for both robots
        user_meaningful_length = find_meaningful_length(user_path['robot_pos'])
        default_meaningful_length = find_meaningful_length(default_path['robot_pos'])
        
        # Use the longer of the two meaningful lengths to show both robots
        meaningful_length = max(user_meaningful_length, default_meaningful_length)
        
        # Ensure we have at least 20 frames for animation
        meaningful_length = max(20, meaningful_length)
        
        # Ensure we don't exceed the actual path lengths
        meaningful_length = min(meaningful_length, len(user_path['robot_pos']), len(default_path['robot_pos']))
        
        # Truncate paths to meaningful length
        user_robot_pos = user_path['robot_pos'][:meaningful_length]
        user_package_pos = user_path['package_pos'][:meaningful_length]
        default_robot_pos = default_path['robot_pos'][:meaningful_length]
        default_package_pos = default_path['package_pos'][:meaningful_length]
        
        # Sample frames for smooth animation (max 150 frames)
        total_frames = len(user_robot_pos)
        if total_frames > 150:
            step = total_frames // 150
            user_robot_pos = user_robot_pos[::step]
            user_package_pos = user_package_pos[::step]
            default_robot_pos = default_robot_pos[::step]
            default_package_pos = default_package_pos[::step]
    
    # Final safety check - ensure all arrays have the same length
    min_frames = min(len(user_robot_pos), len(user_package_pos), len(default_robot_pos), len(default_package_pos))
    if min_frames == 0:
        return go.Figure()
    
    user_robot_pos = user_robot_pos[:min_frames]
    user_package_pos = user_package_pos[:min_frames]
    default_robot_pos = default_robot_pos[:min_frames]
    default_package_pos = default_package_pos[:min_frames]
    
    target_pos = user_path['target_pos']
    
    # Create frames for 2D animation
    frames = []
    for i in range(min_frames):
        # **ZERO SCALING** - Clip all positions to prevent out-of-bounds issues
        user_robot_clipped = clip_position(user_robot_pos[i])
        user_package_clipped = clip_position(user_package_pos[i])
        default_robot_clipped = clip_position(default_robot_pos[i])
        default_package_clipped = clip_position(default_package_pos[i])
        target_clipped = clip_position(target_pos)
        
        frame_data = [
            # User robot
            go.Scatter(
                x=[user_robot_clipped[0]], 
                y=[user_robot_clipped[1]],
                mode='markers', 
                marker=dict(
                    color='lime', 
                    size=20, 
                    symbol='circle',
                    line=dict(width=2, color='darkgreen')
                ), 
                name=f'{user_name}',
                showlegend=(i == 0)
            ),
            # User package
            go.Scatter(
                x=[user_package_clipped[0]], 
                y=[user_package_clipped[1]],
                mode='markers', 
                marker=dict(
                    color='red', 
                    size=15, 
                    symbol='square',
                    line=dict(width=2, color='darkred')
                ), 
                name=f'{user_name} Package',
                showlegend=(i == 0)
            ),
            # Default robot
            go.Scatter(
                x=[default_robot_clipped[0]], 
                y=[default_robot_clipped[1]],
                mode='markers', 
                marker=dict(
                    color='orange', 
                    size=20, 
                    symbol='diamond',
                    line=dict(width=2, color='darkorange')
                ), 
                name=f'{default_name}',
                showlegend=(i == 0)
            ),
            # Default package
            go.Scatter(
                x=[default_package_clipped[0]], 
                y=[default_package_clipped[1]],
                mode='markers', 
                marker=dict(
                    color='darkred', 
                    size=15, 
                    symbol='square',
                    line=dict(width=2, color='red')
                ), 
                name=f'{default_name} Package',
                showlegend=(i == 0)
            ),
            # Target (static)
            go.Scatter(
                x=[target_clipped[0]], 
                y=[target_clipped[1]],
                mode='markers', 
                marker=dict(
                    color='blue', 
                    size=25, 
                    symbol='star',
                    line=dict(width=3, color='darkblue')
                ), 
                name='Target',
                showlegend=(i == 0)
            ),
            # Moving obstacle
            go.Scatter(
                x=[clip_position([user_path['moving_obstacle_pos'][i][0] if i < len(user_path['moving_obstacle_pos']) else 0, user_path['moving_obstacle_pos'][i][1] if i < len(user_path['moving_obstacle_pos']) else -2.0])[0]], 
                y=[clip_position([user_path['moving_obstacle_pos'][i][0] if i < len(user_path['moving_obstacle_pos']) else 0, user_path['moving_obstacle_pos'][i][1] if i < len(user_path['moving_obstacle_pos']) else -2.0])[1]],
                mode='markers', 
                marker=dict(
                    color='red', 
                    size=30,  # Larger size
                    symbol='square',
                    line=dict(width=3, color='darkred')
                ), 
                name='🚨 Moving Obstacle',
                showlegend=(i == 0)
            ),
            # User robot trail
            go.Scatter(
                x=[clip_position(pos)[0] for pos in user_robot_pos[:i+1]], 
                y=[clip_position(pos)[1] for pos in user_robot_pos[:i+1]],
                mode='lines', 
                line=dict(color='lime', width=3, dash='solid'), 
                name=f'{user_name} Path',
                showlegend=(i == 0)
            ),
            # Default robot trail
            go.Scatter(
                x=[clip_position(pos)[0] for pos in default_robot_pos[:i+1]], 
                y=[clip_position(pos)[1] for pos in default_robot_pos[:i+1]],
                mode='lines', 
                line=dict(color='orange', width=3, dash='dot'), 
                name=f'{default_name} Path',
                showlegend=(i == 0)
            ),
            # Warehouse boundaries (larger field)
            go.Scatter(
                x=[-4, 4, 4, -4, -4],
                y=[-4, -4, 4, 4, -4],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name='Warehouse Boundary',
                showlegend=(i == 0)
            ),
            # Static obstacles (shelves) - larger field
            go.Scatter(
                x=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],  # shelf1 (vertical, larger)
                y=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                mode='markers',
                marker=dict(color='gray', size=8, symbol='square'),
                name='Shelf 1',
                showlegend=(i == 0)
            ),
            go.Scatter(
                x=[-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],  # shelf2 (vertical, larger)
                y=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                mode='markers',
                marker=dict(color='gray', size=8, symbol='square'),
                name='Shelf 2',
                showlegend=(i == 0)
            ),
            go.Scatter(
                x=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],  # shelf3 (horizontal, larger)
                y=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                mode='markers',
                marker=dict(color='gray', size=8, symbol='square'),
                name='Shelf 3',
                showlegend=(i == 0)
            )
        ]
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial figure
    path_type = "Full Journey" if show_full_path else "Key Motion"
    fig = go.Figure(
        data=frames[0].data if frames else [],
        layout=go.Layout(
            title=dict(
                text=f"🏆 Robot Competition: {user_name} vs {default_name}<br><sub>({len(frames)} frames - {path_type})</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title="X Position (meters)",
                range=[-4.5, 4.5],
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                # **ZERO SCALING** - Completely lock axes
                fixedrange=True,
                autorange=False,
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                title="Y Position (meters)",
                range=[-4.5, 4.5],
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                # **ZERO SCALING** - Completely lock axes
                fixedrange=True,
                autorange=False,
                tick0=0,
                dtick=1
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            height=700,
            width=900,
            plot_bgcolor='white',
            # **ZERO SCALING** - Additional stability options
            dragmode=False,
            margin=dict(l=50, r=50, t=100, b=50),
            hovermode=False,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="▶️ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 200, "redraw": True},  # Slower animation
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                                # **ZERO SCALING** - Lock axes during animation
                                "relayout": {"xaxis.autorange": False, "yaxis.autorange": False}
                            }]
                        ),
                        dict(
                            label="⏸️ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        ),
                        dict(
                            label="🔄 Reset",
                            method="animate",
                            args=[[frames[0].name], {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue={
                        "font": {"size": 16},
                        "prefix": "Frame:",
                        "visible": True,
                        "xanchor": "right"
                    },
                    transition={"duration": 200, "easing": "cubic-in-out"},
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[[f.name], {
                                "frame": {"duration": 200, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 200}
                            }],
                            label=str(k),
                            method="animate"
                        ) for k, f in enumerate(frames)
                    ]
                )
            ]
        ),
        frames=frames
    )
    
    return fig

def run_evaluation(model_path, env_seed, robot_name="Robot"):
    """Run evaluation and return final observation for robot's senses"""
    if not os.path.exists(model_path):
        return None, True, None, {}
    
    # Evaluation always uses the default environment for a fair race
    env = WarehouseEnv(max_steps=1000)
    obs, _ = env.reset(seed=env_seed)
    model = PPO.load(model_path)
    
    path_data = {
        'robot_pos': [],
        'package_pos': [],
        'target_pos': env.data.body('target').xpos[:2].copy(),  # Only X, Y for 2D
        'moving_obstacle_pos': []
    }
    
    done = False
    steps = 0
    total_reward = 0
    package_picked_up = False
    final_package_to_target_dist = float('inf')
    final_obs = None
    
    # Track movement for debugging
    last_10_distances = []
    movement_stopped_step = None
    
    while not done and steps < 1000:
        # Record positions (only X, Y for 2D)
        robot_pos = env.data.qpos[:2]
        package_pos = env.data.qpos[2:4]
        moving_obstacle_pos = env.data.body('moving_obstacle').xpos[:2]
        
        path_data['robot_pos'].append([robot_pos[0], robot_pos[1]])
        path_data['package_pos'].append([package_pos[0], package_pos[1]])
        path_data['moving_obstacle_pos'].append([moving_obstacle_pos[0], moving_obstacle_pos[1]])
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
        # Store final observation
        final_obs = obs.copy()
        
        if info.get('package_picked_up', False):
            package_picked_up = True
            
        # Track final distance for debugging
        current_dist = info.get('package_to_target_dist', float('inf'))
        final_package_to_target_dist = current_dist
        
        # Track if robot stops moving when close to target
        if package_picked_up and current_dist < 0.5:
            last_10_distances.append(current_dist)
            if len(last_10_distances) > 10:
                last_10_distances.pop(0)
            
            # Check if robot has stopped improving for the last 10 steps
            if len(last_10_distances) >= 10 and steps > 50:
                recent_improvement = max(last_10_distances) - min(last_10_distances)
                if recent_improvement < 0.01 and movement_stopped_step is None:
                    movement_stopped_step = steps
        
        # Early success detection
        if terminated and info.get('is_success', False):
            st.success(f"🎉 {robot_name} completed delivery in {steps} steps!")
            break
    
    success = info.get('is_success', False)
    
    # Debug information
    target_pos = env.data.body('target').xpos[:2]
    final_robot_pos = env.data.qpos[:2]
    final_package_pos = env.data.qpos[2:4]
    
    st.write(f"**🔍 Debug Info for {robot_name}:**")
    st.write(f"- Final robot position: ({final_robot_pos[0]:.3f}, {final_robot_pos[1]:.3f})")
    st.write(f"- Final package position: ({final_package_pos[0]:.3f}, {final_package_pos[1]:.3f})")
    st.write(f"- Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
    st.write(f"- Package to target distance: {final_package_to_target_dist:.3f}")
    st.write(f"- Package picked up: {package_picked_up}")
    st.write(f"- Success threshold: 0.4")
    st.write(f"- Success condition met: {final_package_to_target_dist < 0.4 and package_picked_up}")
    st.write(f"- Total steps: {steps}")
    
    # NEW: Check if robot was terminated for being stuck
    if info.get('severely_stuck', False):
        st.error(f"🚫 Robot was terminated early for being stuck!")
        st.write(f"- Robot barely moved for {env.max_stagnation_steps} steps")
        st.write(f"- This indicates the robot needs more training")
    
    if movement_stopped_step:
        st.write(f"- ⚠️ Robot stopped improving at step: {movement_stopped_step}")
        st.write(f"- Distance when stopped: {last_10_distances[-1]:.3f}")
    
    # Enhanced info with more details
    enhanced_info = {
        'total_reward': total_reward,
        'package_picked_up': package_picked_up,
        'final_robot_to_package_dist': info.get('robot_to_package_dist', 0),
        'final_package_to_target_dist': final_package_to_target_dist,
        'final_obs': final_obs  # NEW: Store final observation for robot's senses
    }
    
    return steps, not success, path_data, enhanced_info

# SIDEBAR - Training Controls
with st.sidebar:
    st.header("🤖 Train Your Robot")
    st.markdown("### 📱 2D View Mode")
    st.info("This version uses 2D visualization for better performance and clarity!")
    
    username = st.text_input("Username", "guest")
    
    with st.expander("⚙️ Training Hyperparameters", expanded=True):
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.0003, format="%.4f")
        total_timesteps = st.slider("Timesteps", 50000, 500000, 200000, step=25000)  # Increased defaults
        n_steps = st.select_slider("N Steps", options=[1024, 2048, 4096], value=2048)
        batch_size = st.select_slider("Batch Size", options=[64, 128, 256], value=128)
        n_epochs = st.slider("N Epochs", 5, 20, 10)
        # NEW: Gamma slider
        gamma = st.slider("Gamma (Discount Factor)", 0.8, 0.999, 0.99, format="%.3f", help="How much the robot values future rewards. Higher is more farsighted.")

    # NEW: Reward Settings Expander
    with st.expander("💰 Reward Settings"):
        st.info("Define the motivation for your robot!")
        success_bonus = st.slider("Success Bonus", 50.0, 500.0, 200.0, help="Big reward for delivering the package.")
        progress_reward_factor = st.slider("Progress Reward Factor", 1.0, 20.0, 5.0, help="Reward for moving the package closer to the target.")
        wall_penalty = st.slider("Wall Penalty", 0.0, 10.0, 2.0, help="Penalty for hitting the outer walls.")
        obstacle_penalty = st.slider("Obstacle Penalty", 0.0, 10.0, 3.0, help="Penalty for hitting the moving obstacle.")
        time_penalty = st.slider("Time Penalty", 0.0, 0.1, 0.005, format="%.4f", help="Small penalty for each step taken.")

    if st.button("🚀 Start Training", type="primary"):
        # UPDATED: Collect all new parameters
        reward_config = {
            'success_bonus': success_bonus,
            'progress_reward_factor': progress_reward_factor,
            'wall_penalty': wall_penalty,
            'obstacle_penalty': obstacle_penalty,
            'time_penalty': time_penalty
        }
        hyperparams = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'total_timesteps': total_timesteps,
            'gamma': gamma,
            'reward_config': reward_config  # Nested dict for rewards
        }
        with st.spinner('Training in progress... This may take a moment.'):
            results = train_user_robot(hyperparams, username)
        st.success(f"✅ Training complete!")
        st.info(f"Model saved: {results['model_path']}")
    
    st.divider()
    
    # Retrain Champion Robot
    if st.button("🏆 Retrain Champion Robot", help="Retrain the champion robot with latest improvements"):
        with st.spinner('Retraining champion robot...'):
            from training import train_champion_robot
            results = train_champion_robot()
        st.success(f"✅ Champion robot retrained!")
        st.info(f"Training time: {results['training_time']:.1f}s")
    
    st.divider()
    
    # NEW: Leaderboard Display
    st.header("🏆 Leaderboard")
    leaderboard_data = load_leaderboard()
    if not leaderboard_data:
        st.info("No scores yet. Train a robot and win a race!")
    else:
        # Display top 5 scores
        for i, (user, steps) in enumerate(list(leaderboard_data.items())[:5]):
            medal = ""
            if i == 0: medal = "🥇"
            elif i == 1: medal = "🥈"
            elif i == 2: medal = "🥉"
            st.markdown(f"**{i+1}. {medal} {user}** - `{steps}` steps")
    
    st.divider()
    
    # Available Models
    st.subheader("📁 Available Models")
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith(".zip")]
        if models:
            for model in models[:5]:  # Show first 5 models
                size_kb = os.path.getsize(f"models/{model}") // 1024
                name = model.replace("_robot_model.zip", "")
                st.text(f"• {name} ({size_kb}KB)")
        else:
            st.warning("No models found")

# MAIN AREA
st.title("🏆 Warehouse Robot Competition (2D View)")

# Competition Section
st.header("⚔️ Start Competition")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    compete_username = st.text_input("Enter your username to compete:", "guest", key="compete_user")

with col2:
    show_full_journey = st.checkbox("🎬 Show Full Journey", value=True, help="Show all steps including when robots get stuck")

with col3:
    st.write("")  # Spacing
    competition_button = st.button("🏁 Start Competition", type="primary", key="start_comp")

if competition_button:
    user_model_path = f"models/{compete_username}_robot_model.zip"
    default_model_path = "models/default_robot_model.zip"
    
    if not os.path.exists(user_model_path):
        st.error(f"❌ No trained model found for '{compete_username}'. Train one first!")
    elif not os.path.exists(default_model_path):
        st.error("❌ Default robot model not found. Please ensure it's trained.")
    else:
        # Store competition data in session state
        competition_seed = np.random.randint(0, 1e9)
        
        # Evaluation Progress
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Running competition...")
            eval_progress = st.progress(0)
            status_text = st.empty()
            
            # Evaluate User Robot
            status_text.text(f"Evaluating {compete_username}'s robot...")
            eval_progress.progress(25)
            user_steps, user_failed, user_path, user_info = run_evaluation(
                user_model_path, competition_seed, f"{compete_username}'s Robot"
            )
            
            # Evaluate Default Robot
            status_text.text("Evaluating default robot...")
            eval_progress.progress(75)
            default_steps, default_failed, default_path, default_info = run_evaluation(
                default_model_path, competition_seed, "Default Robot"
            )
            
            eval_progress.progress(100)
            status_text.text("Competition complete!")
        
        # Clear progress
        progress_container.empty()
        
        # Update leaderboard if user succeeded
        if not user_failed:
            if update_leaderboard(compete_username, user_steps):
                st.balloons()
                st.success(f"🏆 New high score for {compete_username}!")
        
        # Store results in session state for persistence
        st.session_state.competition_results = {
            'user_steps': user_steps,
            'user_failed': user_failed,
            'user_path': user_path,
            'user_info': user_info,
            'default_steps': default_steps,
            'default_failed': default_failed,
            'default_path': default_path,
            'default_info': default_info,
            'username': compete_username,
            'show_full_journey': show_full_journey
        }

# Display Results and Animation if available
if 'competition_results' in st.session_state:
    results = st.session_state.competition_results
    
    # Create two main columns: Results + Animation
    results_col, animation_col = st.columns([1, 2])
    
    with results_col:
        st.subheader("📊 Results")
        
        # User Robot Results
        st.markdown("### 🟢 Your Robot")
        if not results['user_failed']:
            st.success(f"✅ Completed in {results['user_steps']} steps")
        else:
            st.error("❌ Incomplete")
        
        if results['user_info']:
            pickup_icon = "✅" if results['user_info']['package_picked_up'] else "❌"
            st.write(f"📦 Package pickup: {pickup_icon}")
            st.write(f"🎯 Score: {results['user_info']['total_reward']:.0f}")
        
        st.divider()
        
        # Default Robot Results
        st.markdown("### 🟠 Default Robot")
        if not results['default_failed']:
            st.success(f"✅ Completed in {results['default_steps']} steps")
        else:
            st.error("❌ Incomplete")
        
        if results['default_info']:
            pickup_icon = "✅" if results['default_info']['package_picked_up'] else "❌"
            st.write(f"📦 Package pickup: {pickup_icon}")
            st.write(f"🎯 Score: {results['default_info']['total_reward']:.0f}")
        
        st.divider()
        
        # Winner Determination
        st.markdown("### 🏆 Winner")
        if not results['user_failed'] and not results['default_failed']:
            if results['user_steps'] < results['default_steps']:
                st.balloons()
                st.success(f"🎉 **{results['username']} WINS!**")
            elif results['default_steps'] < results['user_steps']:
                st.error("🤖 **Default Robot WINS!**")
            else:
                st.info("🤝 **PERFECT TIE!**")
        elif not results['user_failed']:
            st.balloons()
            st.success(f"🎉 **{results['username']} WINS!**")
        elif not results['default_failed']:
            st.error("🤖 **Default Robot WINS!**")
        else:
            user_perf = results['user_info']['total_reward'] if results['user_info'] else 0
            default_perf = results['default_info']['total_reward'] if results['default_info'] else 0
            if user_perf > default_perf:
                st.warning(f"🥈 **{results['username']} performed better!**")
            elif default_perf > user_perf:
                st.warning("🥉 **Default Robot performed better!**")
            else:
                st.warning("🤝 **DRAW!**")
    
    with animation_col:
        st.subheader("🎬 Competition Replay (2D View)")
        
        if (results['user_path'] and results['default_path'] and 
            results['user_path']['robot_pos'] and results['default_path']['robot_pos']):
            
            animation_fig = create_dual_robot_animation_2d(
                results['user_path'], 
                results['default_path'],
                f"{results['username']}'s Robot",
                "Default Robot",
                results.get('show_full_journey', True)
            )
            st.plotly_chart(animation_fig, use_container_width=True)
            
            # Animation Guide
            with st.expander("💡 2D Animation Guide", expanded=False):
                st.write("🟢 **Green Circle** = Your Robot")
                st.write("🟠 **Orange Diamond** = Default Robot") 
                st.write("🔴 **Red Squares** = Packages")
                st.write("⭐ **Blue Star** = Target")
                st.write("🚨 **Large Red Square** = Moving Obstacle (guards southern routes)")
                st.write("◼️ **Gray Squares** = Static Shelves")
                st.write("📏 **Dashed Gray** = Warehouse Boundary")
                st.write("**Solid Lines** = Robot movement trails")
                st.write("▶️ Click **Play** to watch the competition!")
                st.write("**Game Features:**")
                st.write("• 📱 Faster rendering & stable visualization")
                st.write("• 🎲 Random package/target positions each game")
                st.write("• ⚡ Well-controlled moving obstacle for strategic blocking")
                st.write("• 🛡️ Southern path guardian creates strategic bottleneck")
                st.write("• 📊 Dynamic AI decision-making under extreme pressure")
                st.write("**Note:** Every game is unique with random layouts!")
                st.write("**New Features:**")
                st.write("• ⚡ **Well-Controlled Obstacle** - Optimized speed + near-instant start!")
                st.write("• 🛡️ **Southern Path Blocker** - Guards bottom routes")
                st.write("• 🎲 **Random Start/Goal Positions** - Every game is different!")
                st.write("• 📏 **2x Larger Field** - Longer missions, more strategy")
                st.write("• 🔒 **Zero Scaling** - Completely stable visualization")
                st.write("• 🏃 **Near-Instant Action** - Movement starts by frame 2 (vs 200+ before)!")
                
        else:
            st.warning("⚠️ Animation data not available")
            st.info("Both robots need valid path data for visualization")

    # NEW: Robot's Senses Visualization
    st.subheader("🧠 Robot's Senses (Observation)")
    with st.expander("Click to see what the robot sees", expanded=False):
        obs_vector = results['user_info'].get('final_obs')
        if obs_vector is not None:
            labels = [
                "Robot Pos X", "Robot Pos Y", "Robot Vel X", "Robot Vel Y",
                "Package Pos X", "Package Pos Y", "Target Pos X", "Target Pos Y",
                "Shelf 1 Pos X", "Shelf 1 Pos Y", "Shelf 2 Pos X", "Shelf 2 Pos Y",
                "Shelf 3 Pos X", "Shelf 3 Pos Y", "Obstacle Pos X", "Obstacle Pos Y"
            ]
            
            st.write("**The robot sees the world as 16 numbers:**")
            sense_cols = st.columns(4)
            for i, label in enumerate(labels):
                with sense_cols[i % 4]:
                    # Color code different types of observations
                    if "Robot" in label:
                        st.metric(label=label, value=f"{obs_vector[i]:.2f}", delta="🤖")
                    elif "Package" in label:
                        st.metric(label=label, value=f"{obs_vector[i]:.2f}", delta="📦")
                    elif "Target" in label:
                        st.metric(label=label, value=f"{obs_vector[i]:.2f}", delta="🎯")
                    elif "Shelf" in label:
                        st.metric(label=label, value=f"{obs_vector[i]:.2f}", delta="🧱")
                    elif "Obstacle" in label:
                        st.metric(label=label, value=f"{obs_vector[i]:.2f}", delta="⚠️")
                    else:
                        st.metric(label=label, value=f"{obs_vector[i]:.2f}")
            
            st.info("💡 **How to read:** Position values range from -2 to +2 meters. Velocity shows how fast the robot is moving. The robot uses these 16 numbers to decide its next action!")
        else:
            st.info("No observation data available. Run a competition first!")

else:
    # Show placeholder when no competition has been run
    st.info("👆 Enter a username and click 'Start Competition' to see the 2D robot animation here!")
    
    # Show a demo visualization placeholder
    st.subheader("🎬 2D Competition Animation")
    st.markdown("""
    <div style='text-align: center; padding: 100px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
        <h3>🤖 2D Robot Animation Will Appear Here</h3>
        <p>Start a competition to see both robots competing in fast, clear 2D animation!</p>
        <p><strong>2D Benefits:</strong> Better performance • Clearer view • Faster rendering</p>
    </div>
    """, unsafe_allow_html=True) 