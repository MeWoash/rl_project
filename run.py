import argparse
from collections import defaultdict
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__,'..','src').resolve()))
import PathsConfig as paths_cfg
from ModelTools.Utils import get_all_files, get_last_modified_file
from ModelTools.ModelManager import train_models, run_model
from PostProcessing.PostProcess import generate_all_model_media, generate_models_comparison


def cls():
    try:
        os.system('cls' if os.name=='nt' else 'clear')
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="Model management script")
    parser.add_argument('mode', choices=['train',
                                         'run',
                                         "manual",
                                         'post-process',
                                         'generate',
                                         'compare'],
help="""\
train - train model,
run - run last modified model or model from path if provided with --path,
manual - run manual car parking,
postprocess - run post-process for last edited or at path,
generate - generate mjcf xml file
compare - generates models comparison plots\
""")
    parser.add_argument('--path', help='path to dir/file depending on mode used')
    parser.add_argument('--all', action='store_true', help='do action for all models')

    args = parser.parse_args()

    match args.mode:    
        case 'train':
            train_models()
        case 'run':
            files = get_all_files(paths_cfg.OUT_LEARNING_DIR)
            files_dict = defaultdict(list)
            
            print("======================== Detected model dirs ========================")
            for file in files:
                tmp_path = Path(file)
                directory = tmp_path.parent
                files_dict[directory].append(tmp_path)
            
            
            for i, (key, val) in enumerate(files_dict.items()):
                print(f"[{i}] {key} | n:{len(val)}")
            
            choice_dir = int(input("Choose directory: "))
            key = list(files_dict.keys())[choice_dir]
            
            print("======================== Detected models in dir ========================")
            print(f"{key}:")
            for i, file in enumerate(files_dict[key]):
                print(f"{[i]}: {file.stem}")
                
            choice_model = int(input("Choose number of path to load: "))
            
            model = files_dict[key][choice_model]
            print(f"choice {model}")
            run_model(model)
            
        case "manual":
            import CustomEnvs.manualTestCarParking
            CustomEnvs.manualTestCarParking.main()
        case 'post-process':
            from PostProcessing.PostProcess import generate_model_media
            if args.path:
                last_modified = str(Path(get_last_modified_file(args.path,'.csv'),'..').resolve())
                generate_model_media(last_modified)
            elif args.all:
                generate_all_model_media()  
            else:
                last_modified = str(Path(get_last_modified_file(paths_cfg.OUT_LEARNING_DIR,'.csv'),'..').resolve())
                generate_model_media(last_modified)
        case "generate":
            from MJCFGenerator.Generator import generate_MJCF
            generate_MJCF()
        case "compare":
            generate_models_comparison()
        
if __name__ == '__main__':
    cls()
    main()
