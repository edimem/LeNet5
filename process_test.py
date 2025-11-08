import time
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def test_dataloader_speed():
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])

    dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)

    cpu_cores = os.cpu_count()
    print(f"Detected CPU cores: {cpu_cores}")

    worker_list = [0, 1, 2, 4, 8, 12, 16]
    worker_list = [n for n in worker_list if n <= cpu_cores]

    batch_size = 128
    time_list = []

    print("\nTesting dataloader speed with different num_workers...\n")

    for num in worker_list:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num)
        start = time.time()
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= 100:
                break
        elapsed = time.time() - start
        time_list.append(elapsed)
        print(f"num_workers={num:<2} | Time for 100 batches: {elapsed:.2f} s")

    print("\nTest completed. Observe which num_workers gives the shortest loading time.")

    best_time = min(time_list)
    best_worker = worker_list[np.argmin(time_list)]
    base_time = time_list[0]
    speedup = (base_time - best_time) / base_time * 100

    print("\n========= Analysis =========")
    print(f"Best num_workers: {best_worker}")
    print(f"Shortest time: {best_time:.2f} s")
    print(f"Single-thread time: {base_time:.2f} s")
    print(f"Speed improvement: {speedup:.1f}%")

    if speedup > 10 and best_worker != 0:
        print(f"Recommended num_workers = {best_worker} (significant speedup)")
    else:
        print("Recommended num_workers = 0 (single-thread is faster and more stable)")


if __name__ == "__main__":
    test_dataloader_speed()
