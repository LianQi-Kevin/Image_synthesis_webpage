## 1. Build env
> Server basic configuration \
> RTX3090 + Miniconda + python 3.8 + cuda 11.1

#### Step - 1. Create env
```shell
conda env create -n Taiyi python=3.8
conda activate Taiyi
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge diffusers
pip install gradio==3.1.6 transformers accelerate pytorch-lightning tensorboard zhon
  
```

#### Step - 2. Huggingface login

```shell
huggingface-cli login
```

> https://huggingface.co/docs/huggingface_hub/quick-start
> 
#### Step - 3. Change huggingface cache folder
因为autodl的服务器仅有25GB系统盘，故无法再不扩容的情况下存放模型，故修改huggingface默认的缓存路径
```shell
echo 'export HF_HOME="/root/autodl-tmp/models"' >> /root/.bashrc
source /root/.bashrc
```
