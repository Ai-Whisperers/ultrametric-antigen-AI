# 03 Validation Suite: Computational Scalability

> **Objective**: Prove "Exascale on a Laptop" and define hardware requirements.

## A. Hardware & Scale Matrix (40+ Configs)

**Edge Devices (Batch=1)**

1.  **Raspberry Pi 4**
2.  **Raspberry Pi 5**
3.  **NVIDIA Jetson Nano**
4.  **NVIDIA Jetson Orin**
5.  **iPhone 14 (CoreML)**
6.  **Android Pixel 7 (TFLite)**

**Consumer Laptops (Batch=16)** 7. **MacBook Air (M1)** 8. **MacBook Air (M2)** 9. **MacBook Pro (M1 Pro)** 10. **MacBook Pro (M3 Max)** 11. **Dell XPS 13 (Intel i7)** 12. **Surface Pro 9**

**Workstations (Batch=64)** 13. **NVIDIA RTX 3060** 14. **NVIDIA RTX 3080** 15. **NVIDIA RTX 3090** 16. **NVIDIA RTX 4090** 17. **AMD Radeon RX 7900 XTX**

**Data Center (Batch=512+)** 18. **NVIDIA A100 (40GB)** 19. **NVIDIA A100 (80GB)** (Single Node) 20. **NVIDIA H100** 21. **Google TPU v4** 22. **AWS Inferentia2**

**Distributed Clusters** 23. **2x A100** 24. **4x A100** 25. **8x A100**
...

## B. Speed Benchmarks (Latency)

_Inference time (ms) per genome._

| Model           | 1 Sequence | 1,000 Seqs  | VRAM Usage |
| :-------------- | :--------- | :---------- | :--------- |
| **AlphaFold2**  | 600,000ms  | N/A         | 16GB+      |
| **ESM-1v**      | 1,200ms    | 1,200,000ms | 8GB        |
| **EVE**         | 500ms      | 500,000ms   | 4GB        |
| **Ternary VAE** | **8ms**    | **8,000ms** | **<2GB**   |
