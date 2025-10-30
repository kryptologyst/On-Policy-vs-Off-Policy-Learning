"""Streamlit UI for RL training and visualization."""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import SARSAAgent, QLearningAgent
from envs import make_env, get_env_info
from config_manager import ConfigManager, Config


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Agent Comparison",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning Agent Comparison")
    st.markdown("Compare SARSA (on-policy) vs Q-Learning (off-policy) algorithms")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["FrozenLake-v1", "CartPole-v1", "MountainCar-v0", "GridWorld", "CliffWalking"],
        index=0
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    episodes = st.sidebar.slider("Episodes", 100, 5000, 1000)
    alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.8, 0.99, 0.99, 0.01)
    epsilon = st.sidebar.slider("Initial Epsilon (ε)", 0.01, 0.5, 0.1, 0.01)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 0.999, 0.995, 0.001)
    epsilon_min = st.sidebar.slider("Min Epsilon", 0.001, 0.1, 0.01, 0.001)
    
    # Training button
    if st.sidebar.button("🚀 Start Training", type="primary"):
        train_agents(env_name, episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
    
    # Display environment info
    if st.sidebar.button("ℹ️ Environment Info"):
        try:
            env = make_env(env_name)
            env_info = get_env_info(env)
            st.sidebar.json(env_info)
        except Exception as e:
            st.sidebar.error(f"Error loading environment: {e}")


def train_agents(env_name, episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
    """Train agents and display results."""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create environment
        env = make_env(env_name)
        env_info = get_env_info(env)
        
        # Display environment info
        st.subheader(f"Environment: {env_info['name']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("States", env_info['n_states'])
        with col2:
            st.metric("Actions", env_info['n_actions'])
        with col3:
            st.metric("Episodes", episodes)
        
        # Initialize agents
        agents = {
            'SARSA': SARSAAgent(
                env,
                learning_rate=alpha,
                discount_factor=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            ),
            'Q-Learning': QLearningAgent(
                env,
                learning_rate=alpha,
                discount_factor=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        }
        
        # Training containers
        results = {}
        charts = st.container()
        
        # Train agents
        for agent_name, agent in agents.items():
            status_text.text(f"Training {agent_name}...")
            
            # Initialize metrics storage
            episode_rewards = []
            episode_lengths = []
            epsilon_values = []
            
            # Training loop
            for episode in range(episodes):
                reward, length = agent.train_episode()
                
                episode_rewards.append(reward)
                episode_lengths.append(length)
                epsilon_values.append(agent.epsilon)
                
                # Update progress
                progress = (episode + 1) / episodes
                progress_bar.progress(progress)
                
                # Update epsilon
                agent.decay_epsilon()
            
            results[agent_name] = {
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'epsilons': epsilon_values,
                'value_function': agent.get_value_function(),
                'policy': agent.get_policy()
            }
        
        progress_bar.empty()
        status_text.text("Training completed!")
        
        # Display results
        display_results(results, env_info)
        
    except Exception as e:
        st.error(f"Error during training: {e}")


def display_results(results, env_info):
    """Display training results."""
    
    st.subheader("📊 Training Results")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Final Performance")
        performance_data = []
        for agent_name, data in results.items():
            final_reward = np.mean(data['rewards'][-100:])
            performance_data.append({
                'Agent': agent_name,
                'Final Reward': final_reward,
                'Max Reward': np.max(data['rewards']),
                'Avg Episode Length': np.mean(data['lengths'])
            })
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
    
    with col2:
        st.subheader("Learning Progress")
        # Create learning curves plot
        fig = go.Figure()
        
        for agent_name, data in results.items():
            # Calculate moving average
            window_size = min(50, len(data['rewards']) // 10)
            if window_size > 1:
                moving_avg = np.convolve(data['rewards'], np.ones(window_size)/window_size, mode='valid')
                episodes = np.arange(window_size-1, len(data['rewards']))
                fig.add_trace(go.Scatter(
                    x=episodes,
                    y=moving_avg,
                    mode='lines',
                    name=f"{agent_name} (MA-{window_size})",
                    line=dict(width=2)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=list(range(len(data['rewards']))),
                    y=data['rewards'],
                    mode='lines',
                    name=agent_name,
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Episode",
            yaxis_title="Reward",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Value functions (if applicable)
    if env_info['n_states'] and env_info['n_states'] <= 16:  # Only for small state spaces
        st.subheader("🎯 Value Functions")
        
        # Determine grid shape
        if env_info['n_states'] == 16:
            grid_shape = (4, 4)
        elif env_info['n_states'] == 25:
            grid_shape = (5, 5)
        else:
            grid_shape = None
        
        if grid_shape:
            cols = st.columns(len(results))
            for i, (agent_name, data) in enumerate(results.items()):
                with cols[i]:
                    st.subheader(agent_name)
                    value_grid = data['value_function'].reshape(grid_shape)
                    
                    # Create heatmap
                    fig = px.imshow(
                        value_grid,
                        title=f"{agent_name} Value Function",
                        color_continuous_scale='viridis',
                        aspect='equal'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Policies (if applicable)
    if env_info['n_states'] and env_info['n_states'] <= 16:
        st.subheader("🎮 Policies")
        
        if grid_shape:
            cols = st.columns(len(results))
            action_names = ['Left', 'Down', 'Right', 'Up']  # Default for FrozenLake
            
            for i, (agent_name, data) in enumerate(results.items()):
                with cols[i]:
                    st.subheader(agent_name)
                    policy_grid = data['policy'].reshape(grid_shape)
                    
                    # Create policy heatmap
                    fig = px.imshow(
                        policy_grid,
                        title=f"{agent_name} Policy",
                        color_continuous_scale='tab10',
                        aspect='equal'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Epsilon decay
    st.subheader("📉 Epsilon Decay")
    fig = go.Figure()
    
    for agent_name, data in results.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(data['epsilons']))),
            y=data['epsilons'],
            mode='lines',
            name=agent_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Epsilon Decay",
        xaxis_title="Episode",
        yaxis_title="Epsilon",
        yaxis_type="log",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Results")
    
    for agent_name, data in results.items():
        # Create CSV with results
        df_results = pd.DataFrame({
            'Episode': range(len(data['rewards'])),
            'Reward': data['rewards'],
            'Length': data['lengths'],
            'Epsilon': data['epsilons']
        })
        
        csv = df_results.to_csv(index=False)
        st.download_button(
            label=f"Download {agent_name} Results",
            data=csv,
            file_name=f"{agent_name.lower()}_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
