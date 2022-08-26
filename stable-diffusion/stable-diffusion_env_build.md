## 1. Build env
> Server basic configuration \
> RTX3090 + Miniconda + python 3.8 + cuda 11.3

#### Step - 1. Create env
```shell
wget https://github.com/CompVis/stable-diffusion/archive/refs/heads/main.zip
unzip main.zip && rm main.zip && mv stable-diffusion-main stable-diffusion && cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
pip install gradio
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

> Visit [CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) to download model
> 
> Save model to /root/Image_synthesis_webpage/stable-diffusion/models/


