## min-dalle env Build
> server basic env \
> RTX 3090 + CUDA 11.3 + miniconda 3

### 1. create conda env

> After `conda init`, you must restart your terminal.

```shell
conda create -n min_dalle python=3.8
conda init
conda activate min_dalle
```

### 2. install packages
```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install min-dalle numpy==1.23.0 pillow==9.2.0 requests==2.28.1 gradio
```

### 3. clone
```shell
git clone https://github.com/LianQi-Kevin/Image_synthesis_webpage.git
cd Image_synthesis_webpage/min-dalle
```

### 4. link path
```shell
mkdir -p /root/autodl-tmp/outputs_min-dalle
mkdir -p /root/autodl-tmp/models_min-dalle
ln -s /root/autodl-tmp/outputs_min-dalle /root/Image_synthesis_webpage/min-dalle/outputs
ln -s /root/autodl-tmp/models_min-dalle /root/Image_synthesis_webpage/min-dalle/models
```

### 5. clone min-dalle repository
> for inference only, you don't need to clone `min-dalle` repository, just need to download the models
```shell
wget https://github.com/kuprel/min-dalle/archive/refs/heads/main.zip -O  min_dalle.zip
unzip min_dalle.zip && mv min-dalle-main min_dalle
mkdir -p min_dalle/pretrained/vqgan
mkdir -p min_dalle/pretrained/dalle_bart_mega
```

---

### Models
```
./pretrained/dalle_bart_mega/vocab.json
./pretrained/dalle_bart_mega/merges.txt
./pretrained/dalle_bart_mega/encoder.pt
./pretrained/dalle_bart_mega/decoder.pt
./pretrained/vqgan/detoker.pt

https://huggingface.co/kuprel/min-dalle/resolve/main/vocab.json
https://huggingface.co/kuprel/min-dalle/resolve/main/merges.txt
https://huggingface.co/kuprel/min-dalle/resolve/main/encoder.pt
https://huggingface.co/kuprel/min-dalle/resolve/main/decoder.pt
https://huggingface.co/kuprel/min-dalle/resolve/main/detoker.pt
```