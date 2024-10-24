import torch

# Step 1: Define the function g(X)
def g(X):
    # Example function: g(X) = X^2 + sin(X)
    return X**2 + torch.sin(X)

# Step 2: Prepare input tensor X with shape (12, 2)
batch_size = 12
X = torch.rand(batch_size, 2, requires_grad=True)  # Batch of 12 coordinates in 2D

# Step 3: Compute output
output = g(X)  # Shape will be (12, 2)

# Step 4: Calculate gradients
# Assuming we want to compute gradients of some loss with respect to output
# For demonstration, let's assume we want the sum of outputs as a dummy loss
loss = output.sum()  # This is just an example; replace with actual loss computation

# Backpropagate to compute gradients
loss.backward()

# Access gradients
gradients = X.grad  # Shape will be (12, 2)

print("Output:\n", output)
print("\nGradients:\n", gradients)
