import sys
import time
import statistics
import torch
from neuratal import NeuratalConfig, NeuratalModel

T = int(sys.argv[1])
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[2]]
torch.manual_seed(0)
device = torch.device("cuda")
cfg = NeuratalConfig(vocab_size=262144, hidden_size=512, num_hidden_layers=11,
                     num_attention_heads=8, num_key_value_heads=2,
                     embedding_size=64, intermediate_size=1408,
                     max_position_embeddings=2048)
model = NeuratalModel(cfg).to(device, dtype).eval()
ids = torch.randint(0, 262144, (1, T), generator=torch.Generator().manual_seed(42)).to(device)
with torch.inference_mode():
    for _ in range(5):
        model(ids)
    torch.cuda.synchronize()
    samples = []
    for _ in range(15):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(ids)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
print(f"T={T} {sys.argv[2]} median {statistics.median(samples)*1e3:.2f} ms  min {min(samples)*1e3:.2f}")
