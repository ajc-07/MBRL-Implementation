import torch
import torch.nn as nn

from model import MLP

# Helper functions for training the MLP model based on observations

def train_step(model: MLP, # Neural net
               optimizer: torch.optim.Optimizer, # Usually Adam
               states: torch.Tensor, 
               actions: torch.Tensor,
               next_states: torch.Tensor) -> float:
    """
    Single training step for the MLP dynamics model. Computes predicted deltas, compares with actual deltas, computes loss, and optimizes.

    Args:
        model: The MLP dynamics model.
        optimizer: The optimizer for updating model parameters.
        states: Tensor of current states.
        actions: Tensor of actions taken.
        next_states: Tensor of actual next states.
    
    Returns:
        loss_value: The computed loss value for this training step.
    """
    model.train() # Set to training mode
    optimizer.zero_grad() # Clear gradient cache

    # Compute loss between predicted and actual delta
    predicted_delta = model(states, actions)
    actual_delta = next_states - states
    loss_function = nn.MSELoss()
    loss = loss_function(predicted_delta, actual_delta)
    loss.backward()

    # Update model
    optimizer.step()

    return loss.item() # Returns loss value as a float

# Add mini batches later, simpler for now
def train_model(model: MLP,
                optimizer: torch.optim.Optimizer,
                states: torch.Tensor,
                actions: torch.Tensor,
                next_states: torch.Tensor,
                num_epochs: int = 1000) -> list:
    """
    Trains the MLP dynamics model for a specified number of epochs.

    Args:
        model: The MLP dynamics model.
        optimizer: The optimizer for updating model parameters.
        states: Tensor of current states.
        actions: Tensor of actions taken.
        next_states: Tensor of actual next states.
        num_epochs: Number of training epochs. Default is 1000.
    
    Returns:
        loss_history: List of loss values for each epoch.
    """
    print(f"Training model for {num_epochs} epochs...")
    loss_history = []
    for epoch in range(num_epochs):
        loss_value = train_step(model, optimizer, states, actions, next_states)
        loss_history.append(loss_value)

        # Note epoch + 1 since loop is 0 indexed
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_value:.6f}")

    return loss_history

def evaluate_model(model: MLP,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   next_states: torch.Tensor) -> float:
    """
    Evaluates the MLP dynamics model on a given dataset without updating parameters.

    Args:
        model: The MLP dynamics model.
        states: Tensor of current states.
        actions: Tensor of actions taken.
        next_states: Tensor of actual next states.
    
    Returns:
        loss_value: The computed loss value for the evaluation.
    """
    model.eval() # Set to eval mode
    print("Evaluating model...")
    with torch.no_grad(): # No gradients needed
        predicted_delta = model(states, actions)
        actual_delta = next_states - states
        loss_function = nn.MSELoss()
        loss_value = loss_function(predicted_delta, actual_delta)
    print(f"Evaluation Loss: {loss_value:.6f}")
    return loss_value.item() # Returns loss value as a float