## 1. Build env
> Server basic configuration \
> RTX3090 + Miniconda + python 3.8 + cuda 11.3

#### Step - 1. Create env
```shell
wget https://github.com/Stability-AI/stablediffusion/archive/refs/heads/main.zip
unzip main.zip && rm main.zip && mv stablediffusion-main stable-diffusion && cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
pip install gradio==3.17.0 zhon
```


> `{stable-diffusion}/environment.yaml` need to change the line 20, pytorch-lightning version to 1.5.0

#### Step - 2. link output_dir
```shell
mkdir -p /root/autodl-tmp/outputs
mkdir -p /root/autodl-tmp/models
ln -s /root/Image_synthesis_webpage/stable-diffusion/outputs /root/autodl-tmp/outputs
ln -s /root/Image_synthesis_webpage/stable-diffusion/models /root/autodl-tmp/models
cd /root/Image_synthesis_webpage/stable-diffusion
```

#### Step - 3. Download model

> Visit [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to download the model 'v2-1_512-ema-pruned.ckpt'
> 
> Save model to /root/Image_synthesis_webpage/stable-diffusion/models/


