import argparse
import sys
from pathlib import Path
import subprocess

sys.path.append(str(Path(__file__,'..','..').resolve()))
from PathsConfig import *
from ModelTools.Utils import *
from ModelTools.ModelManager import *



def main():
    parser = argparse.ArgumentParser(description="Model management script")
    parser.add_argument('mode', choices=['train',
                                         'run',
                                         'run_list',
                                         "manual",
                                         'post-process',
                                         'generate'],
help="""\
train - train model,
run - run last modified model or model from path if provided with --path,
run_list - choose model from list,
manual - run manual car parking,
postprocess - run post-process for last edited or at path,
generate - generate mjcf xml file\
""")
    parser.add_argument('--path', help='path to dir/file depending on mode used')

    args = parser.parse_args()

    match args.mode:    
        case 'train':
            train_models()
        case 'run':
            if args.path:
                run_model(args.path)
            else:
                run_model(Path(OUT_LEARNING_DIR))
        case "run_list":
            files = get_all_files(OUT_LEARNING_DIR)
            for key, val in files.items():
                print(f"{[key]}: {val}")
                
            choice = int(input("Choose number of path to load:\n"))
            path = files[choice]
            print(f"choice [{choice}]: {path}")
            run_model(path)   
        case "manual":
            import CustomEnvs.manualTestCarParking
            CustomEnvs.manualTestCarParking.main()
        case 'postprocess':
            from PostProcessing.PostProcess import generate_media_timed
            if args.path:
                last_modified = str(Path(get_last_modified_file(args.path,'.csv'),'..').resolve())
                generate_media_timed(last_modified)
            else:
                last_modified = str(Path(get_last_modified_file(OUT_LEARNING_DIR,'.csv'),'..').resolve())
                generate_media_timed(last_modified)
        case generate:
            from MJCFGenerator.Generator import generate_MJCF
            generate_MJCF()
        
if __name__ == '__main__':
    main()
