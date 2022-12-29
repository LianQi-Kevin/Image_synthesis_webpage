import json
import logging
import os
import time
import argparse
from uuid import uuid4

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything

from utils.prompt_note import prompt_note, examples, parameter_description
from utils.pron_filter import blacklist_filter as ProfanityFilter
from utils.utils import log_set, concat_img


# save images
def save_img(images, prompt, seed, ddim_steps=50, scale=7.5, img_H=512, img_W=512,
             n_samples=4, n_iter=1, ddim_eta=0.0, output_path="./outputs"):
    """
        输出文件结构
        outputs
          └─ "{}_{}_{}".format(global_index, time.strftime("%y-%m-%d_%H-%M-%S"), uuid4()
              ├─ grid.png
              ├─ config.json
              └─ samples
                     ├─ 00001.png
                     ├─ 00002.png
                     ├─ 00003.png
                     └─ 00004.png

        """
    global global_index
    project_uuid = "{}_{}_{}".format(global_index, time.strftime("%y-%m-%d_%H-%M-%S"), uuid4())
    global_index += 1
    logging.info(project_uuid)
    output_dir = os.path.join(output_path, project_uuid)
    sample_path = os.path.join(output_dir, "samples")
    os.makedirs(sample_path, exist_ok=True)
    config_dict = {
        "prompts": prompt,
        "seed": seed,
        "ddim_steps": ddim_steps,
        "ddim_eta": ddim_eta,
        "scale": scale,
        "img_H": img_H,
        "img_W": img_W,
        "n_samples": n_samples,
        "n_iter": n_iter,
        "grid_path": os.path.join(output_dir, "grid.png"),
        "samples_path": list()
    }

    # grid image
    concat_img(images, 2, 512, 512).save(config_dict["grid_path"])
    logging.info("Successful save {}".format(config_dict["grid_path"]))

    # samples image
    for index, img in enumerate(images):
        single_img_path = os.path.join(sample_path, f"{index:05}.png")
        img.save(single_img_path)
        config_dict["samples_path"].append(single_img_path)
        logging.info("Successful save {}".format(single_img_path))

    # config json
    with open(os.path.join(output_dir, "config.json"), "w") as json_f:
        json.dump(config_dict, fp=json_f, ensure_ascii=False, sort_keys=True, indent=4, separators=(",", ": "))

    return output_dir


# pron blacklist words filter
def pron_filter(blacklist_path="utils/pron_blacklist.txt"):
    assert os.path.exists(blacklist_path), "{} not found".format(blacklist_path)
    profanity_filter = ProfanityFilter()
    profanity_filter.add_from_file(blacklist_path)
    return profanity_filter


# use StableDiffusionPipeline generate images
def generate_img(prompt, steps, scale, seed, height=512, width=512, n_samples=4, n_iter=1):
    global pipe_text2img
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={steps}, scale={scale}, img_H={height}, img_W={width}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}")

    seed_everything(seed)
    output = pipe_text2img(prompt, width=width, height=height, guidance_scale=scale,
                           num_inference_steps=steps, num_images_per_prompt=n_samples)
    return output.images, int(seed)


# profanity prompt and save images
def gr_interface(prompt, seed=np.random.randint(1, 2147483646), ddim_steps=50, scale=7.5, img_H=512, img_W=512,
                 random_seed=False, n_samples=4, n_iter=1, ddim_eta=0.0):
    global profanity_filter, args, draw_warning_img

    if random_seed:
        seed = np.random.randint(1, 2147483646)
    else:
        seed = int(seed)

    # check pron_blacklist
    if profanity_filter.is_profane(prompt):
        logging.warning(f"Found pron word in {prompt}")
        print(profanity_filter.censor(prompt))
        return [draw_warning_img, draw_warning_img, draw_warning_img, draw_warning_img], int(seed)
    else:
        # generate images
        images, seed = generate_img(
            prompt=prompt,
            steps=ddim_steps,
            scale=scale,
            seed=seed,
            height=img_H,
            width=img_W,
            n_samples=n_samples,
            n_iter=n_iter
        )
        save_img(images, prompt, seed, ddim_steps, scale, img_H, img_W,
                 n_samples, n_iter, ddim_eta, output_path=args.out_dir)
        return images, int(seed)


# update interactive
def update_interactive(advanced_page):
    if advanced_page:
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


# main page
def gr_advanced_vertical_page():
    global args
    with gr.Blocks(title="109美术高中AI与美术融合课", css="utils/text2img.css") as advanced_app:
        # gr.Column()   垂直      | gr.ROW()  水平
        with gr.Column():
            gr.Markdown("""## 109美术高中AI与美术融合课
                - - -
                """)
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 提示词 - (请勿超过64个词)")
                            prompt_box = gr.Textbox(label="prompts", lines=1, show_label=False)
                            generate_button = gr.Button("开始绘画", elem_id="go_button").style(full_width="True")
                        gr.Markdown("[翻译器](https://www.deepl.com/translator)   [探索提示词](https://openart.ai/)")
                    output_gallery = gr.Gallery(interactive=False).style(grid=[2], height="auto")

                with gr.Column():
                    gr.Markdown("### 高级设置")
                    with gr.Group():
                        with gr.Row():
                            seed_box = gr.Number(label="Seed", value=np.random.randint(1, 2147483646),
                                                 interactive=False, elem_id="seed_box")
                            random_seed_checkbox = gr.Checkbox(label="Random Seed", value=True, interactive=True,
                                                               elem_id="random_seed")
                        with gr.Row():
                            ddim_step_slider = gr.Slider(minimum=10, maximum=50, step=1, value=10, label="Steps",
                                                         interactive=True, visible=args.step_un_show)
                            scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                     label="Guidance Scale", interactive=True)
                            img_H_slider = gr.Slider(minimum=256, maximum=512, step=64, value=512,
                                                     label="Img Height", interactive=True)
                            img_W_slider = gr.Slider(minimum=256, maximum=512, step=64, value=512,
                                                     label="Img Width", interactive=True)
                        gr.Markdown(value=parameter_description)

                    ex = gr.Examples(examples=examples,
                                     inputs=[prompt_box, ddim_step_slider, scale_slider, seed_box],
                                     outputs=[output_gallery, seed_box],
                                     fn=generate_img,
                                     examples_per_page=40,
                                     cache_examples=True)
                    ex.dataset.headers = [""]

            gr.Markdown(prompt_note)

        # style
        prompt_box.style(rounded=(True, True, False, False), container=False)
        generate_button.style(margin=False, rounded=(False, False, True, True), full_width="True")

        # action
        random_seed_checkbox.change(update_interactive,
                                    inputs=[random_seed_checkbox],
                                    outputs=[seed_box])

        prompt_box.submit(gr_interface,
                          inputs=[prompt_box, seed_box, ddim_step_slider, scale_slider, img_H_slider, img_W_slider,
                                  random_seed_checkbox],
                          outputs=[output_gallery, seed_box])

        generate_button.click(gr_interface,
                              inputs=[prompt_box, seed_box, ddim_step_slider, scale_slider, img_H_slider, img_W_slider,
                                      random_seed_checkbox],
                              outputs=[output_gallery, seed_box])

    advanced_app.launch(server_port=6006, share=False, quiet=False, show_error=False, enable_queue=True)


def load_model(model_type="ZH"):
    model_type = model_type.replace('\r', '')
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    if model_type == "ZH":
        model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
    elif model_type == "ZH_EN":
        model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1"
    pipe_text2img = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return pipe_text2img


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser("Taiyi Stable-diffusion TEXT TO IMG")
    parser.add_argument("--out_dir", type=str, help="result output folder", nargs='?',
                        default="/root/Image_synthesis_webpage/Taiyi_StableDiffusion/outputs/")
    parser.add_argument("--model_type", type=str, help="ZH or ZH_EN",
                        choices=["ZH", "ZH_EN"], default="ZH")
    parser.add_argument("--step_un_show", action="store_false", help="whether step option is visible")
    args = parser.parse_args()

    # logging
    log_set(save_level=logging.INFO, show_level=logging.INFO)

    # save index
    global_index = 0

    # kill all old gradio wrap
    gr.close_all()

    # profanity_filter
    profanity_filter = pron_filter("/root/Image_synthesis_webpage/Taiyi_StableDiffusion/utils/pron_blacklist.txt")
    draw_warning_img = Image.open("/root/Image_synthesis_webpage/Taiyi_StableDiffusion/utils/draw_warning.png")

    # create pipeline
    pipe_text2img = load_model(args.model_type)

    # page
    gr_advanced_vertical_page()
