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

#### Step - 2. link output_dir
```shell
mkdir -p /root/autodl-tmp/outputs
mkdir -p /root/autodl-tmp/models
ln -s /root/Image_synthesis_webpage/latent_diffusion/outputs /root/autodl-tmp/outputs
ln -s /root/Image_synthesis_webpage/latent_diffusion/models /root/autodl-tmp/models
```
#### Step - 3. Download model
```shell
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```
