import torch
import time
import numpy as np
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent

def benchmark_actor_critic_recurrent(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Running benchmark on {device.upper()}")    
    # Create an instance of the model
    model = ActorCriticRecurrent(
        num_actor_obs=2557,     # Example observation size
        num_critic_obs=2557,
        num_actions=12,
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        activation="elu",
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    model.eval()    # Test configurations
    batch_sizes = [1, 1000]
    warmup_passes = 10
    timed_passes = 50
    obs_dim = 2557  # must match num_actor_obs    
    for batch_size in batch_sizes:
        model.reset()
        print(f"\n--- Benchmarking with batch size = {batch_size} ---")        
        # Random input data
        observations = torch.randn(batch_size, obs_dim, device=device)
        masks = torch.ones(batch_size, 1, device=device)        
        # Warm-up to stabilize GPU performance
        print("Warming up...")
        for _ in range(warmup_passes):
            with torch.no_grad():
                _ = model.act_inference(observations)        
        # Timing forward passes
        print("Measuring forward-pass time...")
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.perf_counter()        
        with torch.no_grad():
            for _ in range(timed_passes):
                _ = model.act_inference(observations)
            torch.cuda.synchronize() if device == "cuda" else None        
            end_time = time.perf_counter()
        avg_time = (end_time - start_time) / timed_passes        
        print(f"Average forward pass time (batch={batch_size}): {avg_time * 1000:.3f} ms")    
        print("\nBenchmark complete.")
if __name__ == "__main__":
    benchmark_actor_critic_recurrent()