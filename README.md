# DoReFa‑Net on PYNQ‑Z2

Quantized deep learning inference of the DoReFa‑Net convolutional neural network on the Xilinx PYNQ‑Z2 platform, combining low‑bit‑width quantization with FPGA hardware acceleration for real‑time ImageNet classification.

---

## Features

- **1‑bit weight quantization** and **2‑bit activation quantization**  
- Hardware‑accelerated convolutional & fully‑connected layers on Artix‑7 FPGA  
- Seamless Python orchestration via PYNQ’s Overlay API  
- End‑to‑end ImageNet classification Jupyter notebook with performance reporting  

---

## Hardware & Software Requirements

**Hardware**  
- Xilinx PYNQ‑Z2 board (Zynq‑7000 SoC)  
- MicroSD card (≥ 4 GB) with PYNQ image  
- Ethernet cable & USB power cable  
- Host PC for JupyterLab access  

**Software**  
- PYNQ Linux 2.x (with Python 3.6)  
- [qnn](https://github.com/Xilinx/QNN-MO-PYNQ) package (pre‑built overlay + APIs)  
- JupyterLab (provided by PYNQ)  
- Python libraries:  
  - `numpy`  
  - `opencv-python`  
  - `matplotlib`  
  - `Pillow`  
  - `pickle`  

---

