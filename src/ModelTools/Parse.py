import os
import re
from tensorboard.backend.event_processing import event_accumulator



def find_episode_files(path):
    pattern = re.compile(rf'ep-(\d*.)_env-(\d*.)')
    collected_data =[]
    for subdir, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.startswith('events'):
                match = pattern.search(subdir)
            if match:
                # Wyodrębnienie numerów epizodu i środowiska z nazwy folderu
                ep_num, env_num = match.groups()
                # Dodanie informacji do listy
                collected_data.append((int(ep_num), int(env_num), file_name))
    return collected_data

if __name__ == "__main__":
    dir_root = rf"D:\kody\rl_project\out\logs\A2C\\"
    
    episode_files = find_episode_files(dir_root)
    episode_files.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    
    for e in episode_files:
        print(e[0], e[1])