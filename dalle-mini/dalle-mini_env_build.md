## min-dalle env Build
> server basic env \
> RTX 3090 + CUDA 11.2 + Tensorflow 2.5.0 + miniconda 3

### 1. create conda env

> After `conda init`, you must restart your terminal.

```shell
conda create -n dalle_mini python=3.8
conda init
conda activate dalle_mini
```

### 2. Install Packages
```shell
pip install dalle-mini==0.1.0
pip install git+https://github.com/patil-suraj/vqgan-jax.git
pip install -U jaxlib==0.3.14+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
#pip install dalle-mini numpy==1.23.0 pillow==9.2.0 requests==2.28.1 gradio
```

### 3. Link Path
```shell
mkdir -p /root/autodl-tmp/outputs_dalle-mini
ln -s /root/autodl-tmp/outputs_dalle-mini /root/Image_synthesis_webpage/dalle-mini/outputs
```

### 4. Download Model
```shell
cd models
git lfs install
git clone https://huggingface.co/dalle-mini/dalle-mini
git clone https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384
```
