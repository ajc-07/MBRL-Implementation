# MBRL Implementation
## MLP-based, Pure Planning, Model-Based Reinforcement Learning (MBRL) Implementation
The model is trained as specified below:
1. Build MuJoCo Inverted Pendulum-v5 env
2. Collect data from random actions to train model
3. Train dynamics model using MLP that predicts change in state, with MSE.

### Random Action Planner
Steps:
1. Sample random candidate seuqnece, simulate through learned model, score futures, and keep best sequence.
2. Execute first action, repeat.

Run `python main.py --planner simple` to run the simple planner.

Across 50 trials, average cumulative reward of 86.98.

## CEM Planner
What is CEM? Optimizer within MPC to find best sequence of actions given a model.
Motivation: hard to search through all possible action sequences of length H, especially in continuous action spaces
Procedure:
1. Initialize Gaussian distribution over action sequences
2. Sample N candidate action sequences from distribution
3. Evaluate each sequence, calculate predicted cumulative reward from model
4. Update Gaussian using mean and std of top K sequences evaluated
5. Repeat for several iterations, distribution should gradually converge around high reward action sequences

Run `python main.py --planner cem` to run the CEM planner.

Across 50 trials, average cumulative reward of 170.48.

## Online Planning
Continuously train the learned dynamics model on current/online data, instead of only on the initial randomized dataset.

In theory, this should improve the model's accuracy, and thus the planner's performance. Though in our offline experiments, the MLP loss was quite low both in training and eval. The model is already quite accurate, so I don't anticipate much improvement in performance from online training.

But anyways will be implementing this next.
