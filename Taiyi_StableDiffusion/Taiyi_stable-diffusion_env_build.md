## 1. Build env
> Server basic configuration \
> RTX3090 + Miniconda + python 3.8 + cuda 11.1

#### Step - 1. Create env
```shell
conda env create -f environment.yaml
conda activate Taiyi
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

#### Step - 4. Install network tool
```shell
apt update && apt install net-tools lsof
```
