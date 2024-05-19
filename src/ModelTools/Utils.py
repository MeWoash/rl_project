import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__,'..','..').resolve()))
from PathsConfig import *

def get_last_modified_file(directory_path, suffix=".zip"):
    latest_time = 0
    latest_file = None

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(suffix):
                filepath = os.path.join(root, filename)
                if os.path.isfile(filepath):
                    file_mtime = os.path.getmtime(filepath)
                    
                    if file_mtime > latest_time:
                        latest_time = file_mtime
                        latest_file = filepath

    if latest_file:
        print(f"Last modified {suffix} file: {latest_file}")
    else:
        print(f"No {suffix} files found.")
    return latest_file

def get_all_files(directory_path, suffix=".zip"):
    files_dict = {}
    cnt = 1
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(suffix):
                filepath = os.path.join(root, filename)
                if os.path.isfile(filepath):
                    files_dict[cnt] = filepath
                    cnt+=1
                    
    return files_dict


def generate_episodes_summary(df_episodes_all: pd.DataFrame):
    df_episodes_summary = df_episodes_all.groupby(['episode', 'env']).agg(
        learning_step_min=('learning_step', 'min'),
        learning_step_max=('learning_step', 'max'),
        episode_mean_reward=('episode_mean_reward', 'max'),
        episode_mujoco_time=('episode_mujoco_time', 'max'),
        timestamp=('timestamp', 'max')
    ).reset_index()
    df_episodes_summary.sort_values(by="learning_step_max", inplace=True)
    return df_episodes_summary
    
    
def generate_training_stats(df_episodes_all: pd.DataFrame, window = 100):
    df_aggregated = df_episodes_all.groupby(['episode', 'env']).agg(
        learning_step=('learning_step', 'last'),
        timestamp=('timestamp', 'last'),
        episode_mean_reward=('episode_mean_reward', 'last'),
    ).reset_index()
    
    df_aggregated2 = df_aggregated.groupby('learning_step').agg(
        timestamp=('timestamp', 'mean'),
        episode_mean_reward=('episode_mean_reward', 'mean'),
    ).reset_index()
    
    df_rolling_mean = df_aggregated2.rolling(window=window, min_periods=1).mean()
    
    data = {
        "learning_step":df_aggregated2['learning_step'],
        "timestamp": df_aggregated2['timestamp'],
        
        "episode_mean_reward": df_rolling_mean['episode_mean_reward']
    }
    df_training_stats = pd.DataFrame(data)
    return df_training_stats
    
def load_generate_csvs(path_dir:str):
    
    df_episodes_all_path = Path(path_dir, EPISODES_ALL).resolve()
    df_episodes_summary_path = Path(path_dir, EPISODE_STATS).resolve()
    df_training_stats_path = Path(path_dir, TRAINING_STATS).resolve()
    
    
    if not df_episodes_all_path.exists():
        print(f"Did not find {df_episodes_all_path}")
        return None
    else:
        df_episodes_all = pd.read_csv(str(df_episodes_all_path))
    
    if not df_episodes_summary_path.exists():
        df_episodes_summary = generate_episodes_summary(df_episodes_all)
        df_episodes_summary.to_csv(str(df_episodes_summary_path), index=False)
        print(f"Generated {df_episodes_summary_path}")
    else:
        df_episodes_summary = pd.read_csv(str(df_episodes_summary_path))
        
        
    if not df_training_stats_path.exists():
        df_training_stats = generate_training_stats(df_episodes_all)
        df_training_stats.to_csv(str(df_training_stats_path), index=False)
        print(f"Generated {df_training_stats_path}")
    else:
        df_training_stats = pd.read_csv(str(df_training_stats_path))
    
    
    df_episodes_all.set_index(["episode", "env"], inplace=True)
    return df_episodes_all, df_episodes_summary, df_training_stats