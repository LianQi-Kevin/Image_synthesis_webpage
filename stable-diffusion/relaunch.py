import os

# outputs
tempPath = "/root/autodl-tmp/outputs"
if not os.path.exists(tempPath):
    os.mkdir(tempPath)
    print("Successful create {}".format(tempPath))
else:
    print("{} is already exists, {} files in it".format(tempPath, len(os.listdir(tempPath))))

os.system("/root/miniconda3/envs/ldm/bin/python /root/Image_synthesis_webpage/stable-diffusion/text2img_page.py")
