To train agents, go into the `agent` folder and run `diambra run -n 4 python training.py --cfgFile sfiii3n/<config.yaml>`. All the configuration files used in imitation learning and hyperparameter tuning is included in this folder.

For imitation learning, run the jupyter notebook `imitation.ipynb`. 

For Q learning, run `diambra run -n 1 python saving_loading_evaluating.py`. We only use one environment to save memory. Run `test_q.py` to evaluate the Q network.