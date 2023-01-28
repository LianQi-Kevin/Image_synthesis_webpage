out_dir="/root/Image_synthesis_webpage/stable-diffusion/outputs/"
#create output folder
mkdir -p $out_dir
#run
/root/miniconda3/envs/ldm/bin/python /root/Image_synthesis_webpage/stable-diffusion/img2img_page.py --out_dir $out_dir