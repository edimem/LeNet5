###  System Environment
- **OS**: Windows 10 / 11 x64  
- **Python**: 3.8.20  
- **GPU**: NVIDIA (CUDA 11.3, cuDNN 8)  
- **Framework**: PyTorch 1.10.1  

---

###  Main Dependencies

| Module         | Version |
| -------------- | ------- |
| `torch`        | 1.10.1  |
| `torchvision`  | 0.11.2  |
| `torchaudio`   | 0.10.1  |
| `numpy`        | 1.24.4  |
| `pandas`       | 1.2.5   |
| `matplotlib`   | 3.7.2   |
| `scikit-learn` | 1.3.2   |
| `torchsummary` | 1.5.1   |

---

###  Installation

####  Conda Environment (Recommended)
```bash
# Create new environment
conda create -n pytorch python=3.8
conda activate pytorch

# Install core dependencies
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install numpy pandas matplotlib scikit-learn torchsummary