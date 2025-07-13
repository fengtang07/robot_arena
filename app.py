import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO
from environment import WarehouseEnv
from training import train_user_robot

st.set_page_config(
    page_title="Warehouse Robot Competition",
    page_icon="🤖",
    layout="wide"
)

def create_dual_robot_animation(user_path, default_path, user_name="User Robot", default_name="Default Robot", show_full_path=True):
    """Create an animated visualization showing both robots competing"""
    if not user_path or not default_path or not user_path['robot_pos'] or not default_path['robot_pos']:
        return go.Figure()
    
    # Check if moving obstacle data is available
    has_moving_obstacle = ('moving_obstacle_pos' in user_path and 
                          'moving_obstacle_pos' in default_path and 
                          user_path['moving_obstacle_pos'] and 
                          default_path['moving_obstacle_pos'])
    
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
            if has_moving_obstacle:
                user_moving_obstacle_pos = user_path['moving_obstacle_pos'][::step]
                default_moving_obstacle_pos = default_path['moving_obstacle_pos'][::step]
        else:
            # Use all data if reasonable length
            user_robot_pos = user_path['robot_pos'][:min_length]
            user_package_pos = user_path['package_pos'][:min_length]
            default_robot_pos = default_path['robot_pos'][:min_length]
            default_package_pos = default_path['package_pos'][:min_length]
            if has_moving_obstacle:
                user_moving_obstacle_pos = user_path['moving_obstacle_pos'][:min_length]
                default_moving_obstacle_pos = default_path['moving_obstacle_pos'][:min_length]
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
        if has_moving_obstacle:
            user_moving_obstacle_pos = user_path['moving_obstacle_pos'][:meaningful_length]
            default_moving_obstacle_pos = default_path['moving_obstacle_pos'][:meaningful_length]
        
        # Sample frames for smooth animation (max 150 frames)
        total_frames = len(user_robot_pos)
        if total_frames > 150:
            step = total_frames // 150
            user_robot_pos = user_robot_pos[::step]
            user_package_pos = user_package_pos[::step]
            default_robot_pos = default_robot_pos[::step]
            default_package_pos = default_package_pos[::step]
            if has_moving_obstacle:
                user_moving_obstacle_pos = user_moving_obstacle_pos[::step]
                default_moving_obstacle_pos = default_moving_obstacle_pos[::step]
    
    # Final safety check - ensure all arrays have the same length
    arrays_to_check = [user_robot_pos, user_package_pos, default_robot_pos, default_package_pos]
    if has_moving_obstacle:
        arrays_to_check.extend([user_moving_obstacle_pos, default_moving_obstacle_pos])
    
    min_frames = min(len(arr) for arr in arrays_to_check)
    if min_frames == 0:
        return go.Figure()
    
    user_robot_pos = user_robot_pos[:min_frames]
    user_package_pos = user_package_pos[:min_frames]
    default_robot_pos = default_robot_pos[:min_frames]
    default_package_pos = default_package_pos[:min_frames]
    if has_moving_obstacle:
        user_moving_obstacle_pos = user_moving_obstacle_pos[:min_frames]
        default_moving_obstacle_pos = default_moving_obstacle_pos[:min_frames]
    
    target_pos = user_path['target_pos']
    
    # Create frames for animation
    frames = []
    for i in range(min_frames):
        frame_data = [
            # User robot and package
            go.Scatter3d(
                x=[user_robot_pos[i][0]], 
                y=[user_robot_pos[i][1]], 
                z=[user_robot_pos[i][2]],
                mode='markers', 
                marker=dict(color='lime', size=12, symbol='circle'), 
                name=f'{user_name}',
                showlegend=(i == 0)
            ),
            go.Scatter3d(
                x=[user_package_pos[i][0]], 
                y=[user_package_pos[i][1]], 
                z=[user_package_pos[i][2]],
                mode='markers', 
                marker=dict(color='red', size=10, symbol='square'), 
                name=f'{user_name} Package',
                showlegend=(i == 0)
            ),
            # Default robot and package
            go.Scatter3d(
                x=[default_robot_pos[i][0]], 
                y=[default_robot_pos[i][1]], 
                z=[default_robot_pos[i][2]],
                mode='markers', 
                marker=dict(color='orange', size=12, symbol='diamond'), 
                name=f'{default_name}',
                showlegend=(i == 0)
            ),
            go.Scatter3d(
                x=[default_package_pos[i][0]], 
                y=[default_package_pos[i][1]], 
                z=[default_package_pos[i][2]],
                mode='markers', 
                marker=dict(color='darkred', size=10, symbol='square'), 
                name=f'{default_name} Package',
                showlegend=(i == 0)
            ),
            # Target (static)
            go.Scatter3d(
                x=[target_pos[0]], 
                y=[target_pos[1]], 
                z=[target_pos[2]],
                mode='markers', 
                marker=dict(color='blue', size=15, symbol='x'), 
                name='Target',
                showlegend=(i == 0)
            ),
        ]
        
        # Add moving obstacle if available
        if has_moving_obstacle:
            frame_data.append(
                go.Scatter3d(
                    x=[user_moving_obstacle_pos[i][0]], 
                    y=[user_moving_obstacle_pos[i][1]], 
                    z=[user_moving_obstacle_pos[i][2]],
                    mode='markers', 
                    marker=dict(color='darkorange', size=18, symbol='square'), 
                    name='Moving Obstacle',
                    showlegend=(i == 0)
                )
            )
        
        # Add moving obstacle if available
        if has_moving_obstacle:
            frame_data.append(
                go.Scatter3d(
                    x=[user_moving_obstacle_pos[i][0]], 
                    y=[user_moving_obstacle_pos[i][1]], 
                    z=[user_moving_obstacle_pos[i][2]],
                    mode='markers', 
                    marker=dict(color='darkorange', size=18, symbol='square'), 
                    name='Moving Obstacle',
                    showlegend=(i == 0)
                )
            )
        
        # Add robot trails
        frame_data.extend([
            # Robot trails
            go.Scatter3d(
                x=[pos[0] for pos in user_robot_pos[:i+1]], 
                y=[pos[1] for pos in user_robot_pos[:i+1]], 
                z=[pos[2] for pos in user_robot_pos[:i+1]],
                mode='lines', 
                line=dict(color='lime', width=4, dash='solid'), 
                name=f'{user_name} Trail',
                showlegend=(i == 0)
            ),
            go.Scatter3d(
                x=[pos[0] for pos in default_robot_pos[:i+1]], 
                y=[pos[1] for pos in default_robot_pos[:i+1]], 
                z=[pos[2] for pos in default_robot_pos[:i+1]],
                mode='lines', 
                line=dict(color='orange', width=4, dash='dot'), 
                name=f'{default_name} Trail',
                showlegend=(i == 0)
            )
        ])
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial figure
    path_type = "Full Journey" if show_full_path else "Key Motion"
    fig = go.Figure(
        data=frames[0].data if frames else [],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[-4.5, 4.5], title="X (meters)"),
                yaxis=dict(range=[-4.5, 4.5], title="Y (meters)"),
                zaxis=dict(range=[0, 0.3], title="Z (meters)"),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=2.5, y=2.5, z=2.0),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            title_text=f"🏆 Robot Competition: {user_name} vs {default_name} ({len(frames)} frames - {path_type})",
            showlegend=True,
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="▶️ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 50}
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
    if not os.path.exists(model_path):
        return None, True, None, {}
    
    env = WarehouseEnv(max_steps=1000)  # Extended time limit
    obs, _ = env.reset(seed=env_seed)
    model = PPO.load(model_path)
    
    path_data = {
        'robot_pos': [],
        'package_pos': [],
        'target_pos': env.data.body('target').xpos[:3].copy(),
        'moving_obstacle_pos': []
    }
    
    done = False
    steps = 0
    total_reward = 0
    package_picked_up = False
    final_package_to_target_dist = float('inf')
    
    # Track movement for debugging
    last_10_distances = []
    movement_stopped_step = None
    
    while not done and steps < 1000:
        # Record positions
        robot_pos = env.data.qpos[:2]
        package_pos = env.data.qpos[2:4]
        moving_obstacle_pos = env.data.body('moving_obstacle').xpos[:2]
        
        path_data['robot_pos'].append([robot_pos[0], robot_pos[1], 0.05])
        path_data['package_pos'].append([package_pos[0], package_pos[1], 0.05])
        path_data['moving_obstacle_pos'].append([moving_obstacle_pos[0], moving_obstacle_pos[1], 0.1])
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
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
        'final_package_to_target_dist': final_package_to_target_dist
    }
    
    return steps, not success, path_data, enhanced_info

# SIDEBAR - Training Controls
with st.sidebar:
    st.header("🤖 Train Your Robot")
    username = st.text_input("Username", "guest")
    
    with st.expander("⚙️ Training Settings", expanded=False):
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.0003, format="%.4f")
        total_timesteps = st.slider("Timesteps", 50000, 500000, 200000, step=25000)  # Increased defaults
        n_steps = st.select_slider("N Steps", options=[1024, 2048, 4096], value=2048)
        batch_size = st.select_slider("Batch Size", options=[64, 128, 256], value=128)
        n_epochs = st.slider("N Epochs", 5, 20, 10)
    
    if st.button("🚀 Start Training", type="primary"):
        hyperparams = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'total_timesteps': total_timesteps
        }
        with st.spinner('Training in progress...'):
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
st.title("🏆 Warehouse Robot Competition")

# Competition Section
st.header("⚔️ Start Competition")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    compete_username = st.text_input("Enter username to compete with:", "guest", key="compete_user")

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
        st.error("❌ Default robot model not found.")
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
        st.subheader("🎬 Competition Replay")
        
        if (results['user_path'] and results['default_path'] and 
            results['user_path']['robot_pos'] and results['default_path']['robot_pos']):
            
            animation_fig = create_dual_robot_animation(
                results['user_path'], 
                results['default_path'],
                f"{results['username']}'s Robot",
                "Default Robot",
                results.get('show_full_journey', True)  # Default to True if not set
            )
            st.plotly_chart(animation_fig, use_container_width=True)
            
            # Animation Guide
            with st.expander("💡 Animation Guide", expanded=False):
                st.write("🟢 **Green Circle** = Your Robot")
                st.write("🟠 **Orange Diamond** = Default Robot") 
                st.write("🔴 **Red Squares** = Packages")
                st.write("🔵 **Blue X** = Target")
                st.write("**Lines** = Robot movement trails")
                st.write("▶️ Click **Play** to watch the competition!")
                
        else:
            st.warning("⚠️ Animation data not available")
            st.info("Both robots need valid path data for visualization")

else:
    # Show placeholder when no competition has been run
    st.info("👆 Enter a username and click 'Start Competition' to see the dual robot animation here!")
    
    # Show a demo visualization placeholder
    st.subheader("🎬 Competition Animation")
    st.markdown("""
    <div style='text-align: center; padding: 100px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
        <h3>🤖 Dual Robot Animation Will Appear Here</h3>
        <p>Start a competition to see both robots competing in real-time 3D animation!</p>
    </div>
    """, unsafe_allow_html=True)