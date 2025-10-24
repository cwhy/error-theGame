
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, List
import numpy as np


def init_mlp_params(layer_sizes: List[int], key: jnp.ndarray) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Initialize parameters for a multi-layer perceptron.
    
    Args:
        layer_sizes: List of layer sizes, e.g., [784, 128, 10] for input->hidden->output
        key: JAX random key
        
    Returns:
        List of (weight, bias) tuples for each layer
    """
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    
    for i in range(len(layer_sizes) - 1):
        # Xavier/Glorot initialization
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        
        weight_key, bias_key = random.split(keys[i])
        weight = random.normal(weight_key, (fan_in, fan_out)) * std
        bias = random.normal(bias_key, (fan_out,)) * 0.1
        
        params.append((weight, bias))
    
    return params


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU activation function."""
    return jnp.maximum(0, x)


def mlp_forward(params: List[Tuple[jnp.ndarray, jnp.ndarray]], 
                x: jnp.ndarray, 
                activation_fn=relu) -> jnp.ndarray:
    """Forward pass through a multi-layer perceptron.
    
    Args:
        params: List of (weight, bias) tuples for each layer
        x: Input tensor of shape (batch_size, input_dim)
        activation_fn: Activation function to use (default: ReLU)
        
    Returns:
        Output tensor of shape (batch_size, output_dim)
    """
    for i, (weight, bias) in enumerate(params):
        # Linear transformation: x @ weight + bias
        x = jnp.dot(x, weight) + bias
        
        # Apply activation function to all layers except the last one
        if i < len(params) - 1:
            x = activation_fn(x)
    
    return x


def softmax(x: jnp.ndarray) -> jnp.ndarray:
    """Softmax activation function for the output layer."""
    # Subtract max for numerical stability
    x_shifted = x - jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x_shifted)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def mlp_forward_with_softmax(params: List[Tuple[jnp.ndarray, jnp.ndarray]], 
                            x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass with softmax output for classification."""
    # Forward pass through all layers except last
    for i, (weight, bias) in enumerate(params[:-1]):
        x = jnp.dot(x, weight) + bias
        x = relu(x)
    
    # Final layer with softmax
    final_weight, final_bias = params[-1]
    x = jnp.dot(x, final_weight) + final_bias
    return softmax(x)


if __name__ == "__main__":
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Define network architecture: 784 input -> 128 hidden -> 64 hidden -> 10 output
    layer_sizes = [784, 128, 64, 10]
    
    # Initialize parameters
    params = init_mlp_params(layer_sizes, key)
    
    # Create some dummy input data (batch_size=32, input_dim=784)
    batch_size = 32
    input_dim = 784
    dummy_input = random.normal(key, (batch_size, input_dim))
    
    print(f"Network architecture: {layer_sizes}")
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = mlp_forward(params, dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output sample (first 5 values): {output[0, :5]}")
    
    # Forward pass with softmax for classification
    output_softmax = mlp_forward_with_softmax(params, dummy_input)
    print(f"Softmax output shape: {output_softmax.shape}")
    print(f"Softmax output sample (first 5 values): {output_softmax[0, :5]}")
    print(f"Softmax probabilities sum to 1: {jnp.sum(output_softmax[0])}")
    
    # Demonstrate JAX's JIT compilation
    jit_forward = jax.jit(mlp_forward)
    jit_output = jit_forward(params, dummy_input)
    print(f"JIT compiled output shape: {jit_output.shape}")
    
    print("\nNeural network forward pass completed successfully!")
