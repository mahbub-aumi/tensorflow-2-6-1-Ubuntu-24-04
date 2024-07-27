Hey! Before you start the installation process, make sure your video card supports CUDA version 12.1. If it doesn’t, you can still follow the instructions, but be sure to choose the versions carefully. Use the `nvidia-smi` command to check which CUDA version is supported.

Also, here are a video tutorial and some tech support links:

[![YouTube video tutorial](https://img.shields.io/badge/Youtube_video_tutorial-ff3333?style=for-the-badge)](https://www.youtube.com/watch?v=1Tr1ifuSh6o)
[![CUDA & cuDNN support](https://img.shields.io/badge/Cuda_%26_CuDNN_support-33bb33?style=for-the-badge)](https://www.tensorflow.org/install/source#gpu_support_2)
[![TensorRT & TensorFlow support](https://img.shields.io/badge/TensorRT_%26_TensorFlow_support-ffbb33?style=for-the-badge)](https://www.tensorflow.org/install/source#gpu_support_2)

`✅ Step 1` CUDA Installation
----------------------------------------------------------------------------------------------------------------------------------------------------

### :large_blue_circle: Install CUDA:

Open your terminal and type:

```bash
sudo apt update && sudo apt upgrade
sudo apt install build-essential
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

### :large_blue_circle: Add Paths:

Edit the bashrc file:

```bash
nano ~/.bashrc
```

Add these lines at the end of the file:

```bash
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Press Ctrl+O, then ENTER, and finally Ctrl+X to save and exit nano.

Then type:

```bash
source ~/.bashrc
```

Next, create another file:

```bash
sudo nano /etc/ld.so.conf
```

And paste this line into it:

```bash
/usr/local/cuda-12.1/lib64
```

Press Ctrl+O, ENTER, and Ctrl+X to save and exit.

Test the installation:

```bash
sudo ldconfig
echo $PATH
echo $LD_LIBRARY_PATH
sudo ldconfig -p | grep cuda
nvcc --version
```

`✅ Step 2` cuDNN Installation
----------------------------------------------------------------------------------------------------------------------------------------------------

### :large_blue_circle: Install cuDNN

Go to the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive) and download the `cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x` ([direct download link](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz)).

Unzip it:

```bash
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
```

Install cuDNN:

```bash
sudo cp include/cudnn*.h /usr/local/cuda-12.1/include
sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
cd ..
ls -l /usr/local/cuda-12.1/lib64/libcudnn*
```

### :large_blue_circle: Test cuDNN:

Create a test file:

```bash
nano test_cudnn.c
```

Add the following code:

```c
#include <cudnn.h>
#include <stdio.h>

int main() {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        printf("cuDNN successfully initialized.\n");
    } else {
        printf("cuDNN initialization failed.\n");
    }
    cudnnDestroy(handle);
    return 0;
}
```

Press Ctrl+O, ENTER, and Ctrl+X to save and exit.

Compile and run the test:

```bash
gcc -o test_cudnn test_cudnn.c -I/usr/local/cuda-12.1/include -L/usr/local/cuda-12.1/lib64 -lcudnn
./test_cudnn
```

`✅ Step 3` Install TensorRT
----------------------------------------------------------------------------------------------------------------------------------------------------

Visit the [NVIDIA TensorRT download site](https://developer.nvidia.com/tensorrt/download) and download TensorRT 8.6 ([direct link](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz)).

Unzip TensorRT:

```bash
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1
```

Edit your paths again:

```bash
nano ~/.bashrc
```

Add these two lines at the end of the file:

```bash
export PATH=/usr/local/cuda-12.1/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH
```

Press Ctrl+O, ENTER, and Ctrl+X to save and exit.

Then type:

```bash
source ~/.bashrc
```

Fix hard links:

```bash
sudo ldconfig
sudo rm /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn*.so.8
sudo ln -s /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.x.x /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
```

`✅ Step 4` Miniconda Installation
----------------------------------------------------------------------------------------------------------------------------------------------------

Download and install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.4.0-0-Linux-x86_64.sh
bash ./Miniconda3-py310_24.4.0-0-Linux-x86_64.sh
```

Restart the terminal.

`✅ Step 5` Environment Setup
----------------------------------------------------------------------------------------------------------------------------------------------------

Create the environment:

```bash
conda create --name tf_gpu python=3.9
conda activate tf_gpu
```

Install TensorFlow:

```bash
python3 -m pip install tensorflow[and-cuda]
```

Verify the installation:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

To install PyTorch, visit [this link](https://pytorch.org/get-started/locally/), select CUDA version 12.1, and run the command provided.

`✅ Step 6` Making it Work with JupyterLab
----------------------------------------------------------------------------------------------------------------------------------------------------

Open the terminal and run:

```bash
pip install jupyterlab
cd ~
mkdir ml
mkdir tf_gui
jupyter lab
```
