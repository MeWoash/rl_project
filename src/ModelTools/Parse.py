import os
import re
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def find_episode_files(path):
    pattern = re.compile(rf'ep-(\d*.)_env-(\d*.)')
    collected_data =[]
    for subdir, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.startswith('events'):
                match = pattern.search(subdir)
            if match:
                ep_num, env_num = match.groups()
                collected_data.append((int(ep_num), int(env_num), subdir))
    return collected_data

if __name__ == "__main__":
    dir_root = rf"D:\kody\rl_project\out\logs\A2C\A2C_1\\"
    
    episode_files = find_episode_files(dir_root)
    episode_files.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    
    arr = []
    for episode, env, file_path in episode_files:
        ea = event_accumulator.EventAccumulator(file_path)
        ea.Reload()
        scalarsX = ea.Scalars('episode/pos_X')
        scalarsY = ea.Scalars('episode/pos_Y')
        xArr = []
        yArr = []
        for scalar in scalarsX:
            xArr.append(scalar.value)
        for scalar in scalarsY:
            yArr.append(scalar.value)
            
        arr.append([xArr,yArr])
    
    
    for a in arr:
        awd = np.array(a)
        x_y = awd.T
        plt.plot(x_y[:,1], x_y[:,0])

    plt.scatter(5, 5, s=30, c='r')
    plt.grid()
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.gca().invert_xaxis()
    plt.show()