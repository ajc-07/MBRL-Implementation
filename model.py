import numpy as np
import torch
import torch.nn as nn

# https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
# Two objects, a cart and a pole
# Action space: Box(-3.0, 3.0, (1,), float32)
#   Actions have shape (1,), scalar, value is a float between -3 and 3, representing forward/backward force applied to the cart.
# Observation space: Box(-inf, inf, (4,), float64)
#   Observations have shape (4,), representing the cart's position, pole's angle, cart's velocity, and pole's angular velocity.

class MLP(nn.Module):
    """
    MLP-based dynamics model for MBRL. Takes in current state and action, outputs predicted next state.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int):
        """
        Initialize MLP model with given sizes, and randomly initialize parameters.

        Args:
            input_dim (int): Dimension of input (state + action).
            hidden_dim (int): Dimension of hidden layer.
            output_dim (int): Dimension of output (predicted next state).
        
        Returns:
            None
        """
        super().__init__() # initialize the neural network module class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize random weights and biases for simple 2 layer MLP
        # Note we scale the weights by 0.01 to prevent large initial values
        self.W1 = nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.W1.bias)

        self.W2 = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.normal_(self.W2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.W2.bias)

        self.relu = nn.ReLU()

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Predicts change from current state to next state given current state and action using the MLP model. Note state_dim and output_dim are the same (4,) while action_dim is (1,).
        Learning the change in state can save us from relearning the current state as part of its prediction. Also they do this in the reference project: https://github.com/thubZ09/model-based-reinforcement-learning
    
        Args: 
            state: tensor of shape (state_dim)
            action: tensor of shape (action_dim)
        
        Returns:
            predicted_state: tensor of shape (output_dim)
        """
        # Concatenate state and action as input to MLP
        # Note axis = -1 means concatenate at highest dimension, in this case both are 1D, so result is a 1D array
        X = torch.cat([state, action], dim=-1)  # shape (state_dim + action_dim,) = (input_dim,)

        # Forward pass through MLP
        # First layer
        # X: (1, input_dim)
        # W1: (input_dim, hidden_dim)
        Z1 = self.W1(X)  # shape (1, hidden_dim)
        # Apply ReLU activation
        A1 = self.relu(Z1)  # shape (1, hidden_dim)

        # Second layer
        # A1: (1, hidden_dim)
        # W2: (hidden_dim, output_dim)
        Z2 = self.W2(A1)  # shape (1, output_dim)
        delta = Z2  # shape (output_dim,)
        return delta  # return predicted change in state, not absolute next state

    def predict_next_state(self,
                           state: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
        """
        Helper function to return next state instead of just delta.

        Args: 
            state: tensor of shape (state_dim)
            action: tensor of shape (action_dim)

        Returns:   
            next_state: tensor of shape (output_dim)
        """
        delta = self.forward(state, action)  # shape (output_dim,)
        next_state = state + delta  # shape (output_dim,)
        return next_state
