# Self-Pruning Neural Network

## Idea
This model learns to remove its own weights during training.

Each weight has a gate:
weight * sigmoid(gate)

If gate becomes 0, weight is removed.

## Loss Function
Total Loss = CrossEntropy + λ * Sparsity Loss

Sparsity Loss = sum of all gate values (L1 penalty)

## Experiment
λ = 0.001

Due to time constraints, limited training was performed.

## Conclusion
Higher λ increases sparsity but may reduce accuracy.
