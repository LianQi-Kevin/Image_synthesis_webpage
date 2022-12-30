out_dir="/root/Image_synthesis_webpage/Taiyi_StableDiffusion/outputs/"
model_type="ZH"
#create output folder
mkdir -p $out_dir
#run
/root/miniconda3/envs/Taiyi/bin/python /root/Image_synthesis_webpage/Taiyi_StableDiffusion/text2img_page.py \
--out_dir $out_dir \
--model_type $model_type