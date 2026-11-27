# MBRL Implementation
## MLP-based, Pure Planning, Model-Based Reinforcement Learning (MBRL) Implementation
The model is trained as specified below:
1. Build MuJoCo Inverted Pendulum-v5 env
2. Collect data from random actions to train model
3. Train dynamics model using MLP that predicts change in state, with MSE.

### Simple Random Shooting Planner
Steps:
1. Sample random candidate sequence, simulate through learned model, score futures, and keep best sequence.
2. Execute first action, repeat.

Single run command: 
```
uv run python main.py --planner simple --verbose
```

Across 50 trials, this method yields an average cumulative reward of 67.14.
Reproduce with 
```
uv run python main.py --planner simple --num_trials 50
```

### CEM Planner
What is CEM? Optimizer within MPC to find best sequence of actions given a model.
Motivation: hard to search through all possible action sequences of length H, especially in continuous action spaces
Procedure:
1. Initialize Gaussian distribution over action sequences
2. Sample N candidate action sequences from distribution
3. Evaluate each sequence, calculate predicted cumulative reward from model
4. Update Gaussian using mean and std of top K sequences evaluated
5. Repeat for several iterations, distribution should gradually converge around high reward action sequences

Single run command: 
```
uv run python main.py --planner cem --verbose
```

Across 50 trials, this method yields an average cumulative reward of 96.96.
```
uv run python main.py --planner cem --num_trials 50
```

### Online Learning
Continuously train the learned dynamics model on current/online data, instead of only on the initial randomized dataset.

In theory, this should improve the model's accuracy, and thus the planner's performance. Though in our offline experiments, the MLP loss was quite low both in training and eval. The model is already quite accurate, so I don't anticipate much improvement in performance from online training.

But anyways will be implementing this next.

Procedure:
1. Pretrain the MLP on random data as before
2. Run planner
3. After each step, train the model on the newly observed (state, action, next_state) transition
4. Update model parameters, and continue to next step using new updated model

Note that model's online learning should not carry across trials, so we reset the model after each trial.

50 trial, online learning, simple random shooting: 84.08
```
uv run python main.py --planner simple --num_trials 50 --online_learning
```

50 trial, online learning, CEM: 92.32
```
uv run python main.py --planner cem --num_trials 50 --online_learning
```

### Conclusion/Thoughts (So Far)
So far I've implemented a simple MLP-based dynamics model and two planners: a simple random shooting planner and a CEM planner. The CEM planner performs better than the simple random shooting planner, as expected. Online learning does not seem to have a significant impact on performance: the difference in average cumulative reward is small enough to be chalked up to variance. I believe this is likely because the model is already quite accurate after pretraining on random data, since the system itself is so simple, as evidenced by the low MSE loss on both training and evaluation data.

Next steps:
- Read JEPA paper as suggested, and try to implement.
- Read https://arxiv.org/pdf/1906.09237, and again try to implement.

## JEPA-based MBRL!!
Yet to begin, reading first! Will be taking notes in Google doc.