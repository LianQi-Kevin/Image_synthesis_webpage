### [./latent-Diffusion-Models](./latent-Diffusion-Models) fork from [Latent Diffusion Models](https://github.com/pesser/stable-diffusion) 

---

## 1. Build env
> Server basic configuration \
> RTX3090 + Miniconda + python 3.8 + cuda 11.3

#### Step - 1. Create env
```shell
git clone https://github.com/LianQi-Kevin/LD_web_page.git
cd LD_web_page/latent-Diffusion-Models
conda env create -f environment.yaml
conda activate ldm
```

#### Step - 2. Download model
```shell
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

#### Step - 3. Sample
```shell
python scripts/txt2img.py \
  --prompt "a virus monster is playing guitar, oil on canvas" \
  --config configs/latent-diffusion/txt2img-1p4B-eval.yaml \
  --ckpt models/ldm/text2img-large/model.ckpt \
  --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```