import torch
from src.cluster_trading.system import CoopTradingSystem

def verify():
    print("=== Verifying Cluster-Based Trading System ===")

    # 1. Setup Dummy Data
    B, N, T = 2, 50, 200
    prices = torch.rand(B, N, T) * 100 + 10  # Random prices around 60
    prices.requires_grad = True # Check if we can backprop to input (symbolic)

    print(f"Input Prices Shape: {prices.shape}")

    # 2. Init System
    system = CoopTradingSystem(n_clusters=5, feature_window=10, temp=0.5)
    print("System Initialized.")

    # 3. Init Centroids (Data Driven)
    system.initialize_clusters(prices)

    # 4. Forward Pass (Training Mode - Soft)
    print("\n--- Forward Pass (Soft) ---")
    out = system(prices, hard=False)

    features = out['features']
    assignments = out['assignments']
    positions = out['positions']

    print(f"Features Shape: {features.shape} (Expected: {B}, {N}, {T-9}, 10)") # Window 10
    print(f"Assignments Shape: {assignments.shape} (Expected: {B}, {N}, {T-9}, 5)")
    print(f"Positions Shape: {positions.shape} (Expected: {B}, {N}, {T-9})")

    # 5. Check Gradients
    print("\n--- Backward Pass Check ---")
    # Loss = Maximize Mean Position (Dummy)
    loss = -positions.mean()
    loss.backward()

    print(f"Loss: {loss.item()}")

    if system.cluster_layer.centroids.grad is not None:
        print("Centroids Gradient: OK")
    else:
        print("Centroids Gradient: MISSING!")

    if prices.grad is not None:
        print("Prices Gradient: OK (End-to-End Differentiable)")
    else:
        print("Prices Gradient: MISSING!")

    # 6. Forward Pass (Hard Mode)
    print("\n--- Forward Pass (Hard) ---")
    with torch.no_grad():
        out_hard = system(prices, hard=True)
        print("Positions (Hard):", out_hard['positions'].shape)
        # Check if assignments are one-hot
        assigns = out_hard['assignments']
        is_one_hot = (assigns.sum(dim=-1) - 1.0).abs().max() < 1e-5
        print(f"Assignments are One-Hot: {is_one_hot}")

if __name__ == "__main__":
    verify()
