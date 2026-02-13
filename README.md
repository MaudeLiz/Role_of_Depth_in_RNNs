# On the Role of Depth in the Expressivity of RNNs

## Installation
This project uses Python 3.9. 
It is recommended to run this code in a virtual environment. I used `venv` for this project.
To setup the virtual environment and download all the necessary packages, follow the steps below
First, load the Python module you want to use:
```
module load python/3.9
```
Or use `python3.9` instead of `python` in the following command that creates a virtual environment in your home directory:
```
python -m venv $HOME/<env>
```
Where `<env>` is the name of your environment. Finally, activate the environment:
```
source $HOME/<env>/bin/activate
```
Now to install the packages simply run
```
pip install -r requirements.txt
```

## Configuration
Configuration files can be found in configs directory. Make sure to update the `logger` to log on wandb and `save_dir` entries.
### Experiment
There are two main configurations, one for synthetic experiements and the other for the shakespeare dataset.
Different type of synthetic experiments are determined by the `copymode` entry in the configuration files. Their implementation can be found in `src/data_synthetic.py`.
### Models
There are three models configurations: rnn, cprnn, s4. Their parameterization is set in `configs/model/`. Implementation of the models can be found in `src/models.py`.

## Run
Specify the configaration file (config_name) to use in `src/train.py`. 
Then run 
```
python src/train.py
```
Alternatively, you can use the bash script 
```
src/script/run.sh
```
which automatically activates the virtual environment.  
