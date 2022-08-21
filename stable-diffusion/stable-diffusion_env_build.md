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
conda install -c conda-forge diffusers
```

#### Step - 2. link output_dir
```shell
mkdir -p /root/autodl-tmp/outputs_stable_diffusion
mkdir -p /root/autodl-tmp/models
ln -s /root/Image_synthesis_webpage/stable_diffusion/outputs /root/autodl-tmp/outputs_stable_diffusion
ln -s /root/Image_synthesis_webpage/stable_diffusion/models /root/autodl-tmp/models
```

[comment]: <> (#### Step - 3. Download model)

[comment]: <> (```shell)

[comment]: <> (···)
