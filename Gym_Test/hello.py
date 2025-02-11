import torch
import time

# Check if ROCm is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a large model
class StressTestModel(torch.nn.Module):
    def __init__(self):
        super(StressTestModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(13000, 13000),
            torch.nn.ReLU(),
            torch.nn.Linear(13000, 13000),
            torch.nn.ReLU(),
            torch.nn.Linear(13000, 13000),
            torch.nn.ReLU(),
            torch.nn.Linear(13000, 13000),
        )
    
    def forward(self, x):
        return self.layers(x)

# Create the model and move it to the GPU
model = StressTestModel().to(device)

# Generate random data for input and target
input_data = torch.randn(1024, 13000, device=device)
target = torch.randn(1024, 13000, device=device)

# Define a loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Run a few iterations of forward and backward passes
print("Starting stress test...")
start_time = time.time()
num_iterations = 10

for iteration in range(num_iterations):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}")

end_time = time.time()
print(f"Stress test completed in {end_time - start_time:.2f} seconds.")

# GPU Memory Usage
if torch.cuda.is_available():
    print(f"Peak GPU Memory Allocated: {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
    print(f"Peak GPU Memory Cached: {torch.cuda.max_memory_reserved(device) / (1024**2):.2f} MB")
