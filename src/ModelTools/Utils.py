import datetime
import sys
import json
from pathlib import Path
import uuid

sys.path.append(str(Path(__file__,'..','..').resolve()))
from PathsConfig import *
from CustomEnvs import CarParkingEnv
from ModelTools.Callbacks import *

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