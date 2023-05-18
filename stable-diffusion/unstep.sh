out_dir="/root/Image_synthesis_webpage/stable-diffusion/outputs/"
ckpt_path="/root/autodl-fs/models/StableDiffusion/realisticVisionV20_v20.safetensors"
config_path="/root/Image_synthesis_webpage/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"

#create export folder
mkdir -p "/root/autodl-tmp/outputs"

#run
/root/miniconda3/envs/ldm/bin/python /root/Image_synthesis_webpage/stable-diffusion/tab_page.py \
--out_dir $out_dir \
--ckpt $ckpt_path \
--config $config_path \
--step_un_show