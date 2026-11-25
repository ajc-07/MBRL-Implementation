### Simple MBRL
Specifically pure planning version of Model-Based RL:

What is CEM? Optimizer within MPC to find best sequence of actions given a model.
Motivation: hard to search through all possible action sequences of length H, especially in continuous action spaces
Procedure:
1. Initialize Gaussian distribution over action sequences
2. Sample N candidate action sequences from distribution
3. Evaluate each sequence, calculate predicted cumulative reward from model
4. Update Gaussian using mean and std of top K sequences evaluated
5. Repeat for several iterations, distribution should gradually converge around high reward action sequences

Steps:
1. Build MuJoCo Inverted Pendulum-v5 env
2. Collect data from random actions to train model
3. Train dynamics model using MLP that predicts change in state, with MSE.
4. Start with simple planner:
5. Sample random candidate seuqnece, simulate through learned model, score futures, and keep best sequence.
6. Execute first action, repeat.
7. Upgrade to CEM
8. Make it online, let MPC collect more real transitions.

Components:
- Build MuJoCO env
- Trained dynamics model
- Planner to simulate candidate action sequences, score, and return first action of the best one
