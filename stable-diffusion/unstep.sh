out_dir="/root/Image_synthesis_webpage/stable-diffusion/outputs/"
ckpt_path="/root/Image_synthesis_webpage/stable-diffusion/models/v2-1_512-ema-pruned.ckpt"
#create output folder
mkdir -p $out_dir
#create model folder
mkdir -p "/root/autodl-tmp/models"
mkdir -p "/root/autodl-tmp/outputs"
#if model exits
if [ ! -f "$ckpt_path" ]; then
cp /root/autodl-nas/models/StableDiffusion/v2-1_512-ema-pruned.ckpt $ckpt_path
fi
#run
/root/miniconda3/envs/ldm/bin/python /root/Image_synthesis_webpage/stable-diffusion/text2img_page.py \
--out_dir $out_dir \
--ckpt $ckpt_path \
--step_un_show