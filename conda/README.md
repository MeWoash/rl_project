# To install workspace do following steps:
1. Install Anaconda
2. Open anaconda prompt
3. create environemnt from environemnt.yaml:
    ```
    conda env create --file <path_to_dir>\environment.yml
    ``````

# Update environment if env file changed:
1. Open anaconda prompt:
    ```
    conda env update --file <path_to_dir>\environment.yml --prune
    ```

# Remove environment
1. Open anaconda prompt:
    ```
    conda env remove --name env_name
    ```

# Share environment
1. cross-platform
    ```
    conda env export --from-history > environment.yml
    ```
2. system specific
    ```
    conda env export > environment.yml
    ```

# Tensorboard
1. 
    ```
    tensorboard --logdir <logs_path>
    ```