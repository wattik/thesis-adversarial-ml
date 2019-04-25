#Adversarial Machine Learning in Security

This live repo contains sources and text created solely for the purposes of my master thesis. 

The goal of the thesis is to design an activity detector that is able to distinguish normal 
users from malicious adversaries. This is done by connecting the standard Empirical Risk Minimisation framework 
with a Game-Theoretic model of the attacker.

The data used in training and evaluation were provided by Trend Micro. Unfortunately, since the dataset contains
 private information about real users it is not published in the repo nor it will be made publicly available later on. 

To run the experiment, use file `sources/experiment.py`. The current model is 
stored in `sources/models/pytorch/langrange_net.py`. 

The training manager is in `sources/models/pytorch/langrange_trainer.py`.

The model of the attacker is located in `sources/threat_model/histogram_attacker.py`. 

 