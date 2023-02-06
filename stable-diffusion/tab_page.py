import json
import logging
import os
import time
import argparse
import shutil
from uuid import uuid4

import gradio as gr
import numpy as np
from PIL import Image

from utils.all2img import all2img
from utils.prompt_note import parameter_description_img2img, parameter_description, title_msg
from utils.prompt_note import prompt_note, end_message, examples
from utils.pron_filter import blacklist_filter as ProfanityFilter
from utils.utils import log_set, concat_img, clear_port


# build save path and save images(grid_image, init_image, samples)
def save_img(images: list, prompt: str, seed: int, ddim_steps: int, scale: float, strength: float = None,
             init_img_path: str = None, img_H=512, img_W=512, n_samples=4, n_iter=4,
             ddim_eta: float = 1.0, output_path="outputs"):
    """
        'export files'
        outputs
            └─ "{}_{}_{}".format(global_index, time.strftime("%y-%m-%d_%H-%M-%S"), uuid4()
                ├─ grid.png
                ├─ init_image.png
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

    # add img2img msg
    if init_img_path is not None and strength is not None:
        config_dict["strength"] = strength
        config_dict["init_img_path"] = os.path.join(output_dir, "init_image.png")

        # check init img exists
        if os.path.exists(init_img_path):
            shutil.copy(init_img_path, config_dict["init_img_path"])
            logging.info("Successful save {}".format(config_dict["init_img_path"]))

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


# pron prompt filter
def pron_filter(blacklist_path="utils/pron_blacklist.txt"):
    assert os.path.exists(blacklist_path), "{} not found".format(blacklist_path)
    # read blacklist
    profanity_filter = ProfanityFilter()
    profanity_filter.add_from_file(blacklist_path)
    return profanity_filter


# update 'random seed' button interactive value
def update_interactive(advanced_page):
    return gr.update(interactive=not advanced_page)


# update 'Image source' button visible value, between 'upload' and 'canvas'
def update_visible(switch_button):
    if switch_button == "switch to upload":
        return gr.update(visible=True), gr.update(visible=False), gr.update(value="switch to canvas")
    elif switch_button == "switch to canvas":
        return gr.update(visible=False), gr.update(visible=True), gr.update(value="switch to upload")


def gr_interface_un_save(prompt, ddim_steps=50, scale=7.5, seed=1024, img_H=512, img_W=512,
                         n_samples=4, n_iter=1, ddim_eta=0.0):
    global all2img

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={ddim_steps}, scale={scale}, img_H={img_H}, img_W={img_W}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}, ddim_eta={ddim_eta}")

    all_samples = all2img.text2img(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter),
                                   img_H=img_H, img_W=img_W, ddim_steps=int(ddim_steps), scale=scale,
                                   ddim_eta=ddim_eta)
    images = all2img.postprocess(all_samples, single=True)

    return images, int(seed)


def text2img_infer(prompt, seed=np.random.randint(1, 2147483646), ddim_steps=50, scale=7.5, img_H=512, img_W=512,
                   random_seed=False, n_samples=4, n_iter=1, ddim_eta=0.0):
    global profanity_filter, draw_warning_img, args, all2img

    if random_seed:
        seed = np.random.randint(1, 2147483646)
    else:
        seed = int(seed)

    # print logging
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={ddim_steps}, scale={scale}, img_H={img_H}, img_W={img_W}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}, ddim_eta={ddim_eta}")

    # check pron_blacklist
    if profanity_filter.is_profane(prompt):
        logging.warning(f"Found pron word in {prompt}")
        print(profanity_filter.censor(prompt))
        return [draw_warning_img], int(seed)

    # synthesis
    all_samples = all2img.text2img(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter),
                                   img_H=img_H, img_W=img_W, ddim_steps=int(ddim_steps), scale=scale,
                                   ddim_eta=ddim_eta)

    # postprocess
    images = all2img.postprocess(all_samples, single=True)

    # save
    save_img(images=images, prompt=prompt, seed=seed, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
             scale=scale, img_H=img_H, img_W=img_W, n_samples=n_samples, n_iter=n_iter, output_path=args.out_dir)
    return images, int(seed)


# def img2img_infer():
def img2img_infer(prompt, init_img_path, canvas_init_path, seed=np.random.randint(1, 2147483646), ddim_steps=50,
                  strength=0.8, scale=7.5, img_H=512, img_W=512, random_seed=True,
                  n_samples=4, n_iter=1, ddim_eta=0.0):
    global all2img, profanity_filter, draw_warning_img, args, init_warning_img

    if random_seed:
        seed = np.random.randint(1, 2147483646)
    else:
        seed = int(seed)

    # init_image
    if init_img_path is None and canvas_init_path is None:
        return [init_warning_img], int(seed)
    elif init_img_path is None and canvas_init_path is not None:
        init_img_path = canvas_init_path
    elif init_img_path is not None and canvas_init_path is None:
        init_img_path = init_img_path

    # print logging
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={ddim_steps}, scale={scale}, img_H={img_H}, img_W={img_W}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}, ddim_eta={ddim_eta}, strength={strength}")
    logging.info(f"init_img_path={init_img_path}")

    # check pron_blacklist
    if profanity_filter.is_profane(prompt):
        logging.warning(f"Found pron word in {prompt}")
        return [draw_warning_img], int(seed)

    # synthesis
    all_samples = all2img.img2img(prompt=prompt, init_img=init_img_path, seed=seed, n_samples=n_samples, n_iter=n_iter,
                                  ddim_steps=ddim_steps, scale=scale, ddim_eta=ddim_eta, strength=strength, plms=False)
    # postprocess
    images = all2img.postprocess(all_samples, single=True)

    # save
    save_img(images=images, init_img_path=init_img_path, prompt=prompt, strength=strength, seed=seed,
             ddim_steps=ddim_steps, scale=scale, img_H=img_H, img_W=img_W, n_samples=n_samples,
             n_iter=n_iter, ddim_eta=ddim_eta, output_path=args.out_dir)
    return images, int(seed)


def gr_advanced_vertical_page():
    global args

    with gr.Blocks(title="109美术高中AI与美术融合课", css="utils/text2img.css") as advanced_app:
        gr.Markdown(title_msg)
        with gr.Tab("Text to Img"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("#### 提示词 - (请勿超过64个词)")
                                prompt_box = gr.Textbox(label="prompts", lines=1, show_label=False)
                                generate_button = gr.Button("开始绘画", elem_id="go_button").style(full_width="True")
                            gr.Markdown("[翻译器](https://www.deepl.com/translator)   [探索提示词](https://openart.ai/)")
                        output_gallery = gr.Gallery(interactive=False).style(grid=2, height="1024px")

                    with gr.Column():
                        gr.Markdown("### 高级设置")
                        with gr.Group():
                            with gr.Row():
                                seed_box = gr.Number(label="Seed", value=np.random.randint(1, 2147483646),
                                                     interactive=False,
                                                     elem_id="seed_box")
                                random_seed_checkbox = gr.Checkbox(label="Random Seed", value=True, interactive=True,
                                                                   elem_id="random_seed")
                            with gr.Row():
                                ddim_step_slider = gr.Slider(minimum=10, maximum=50, step=1, value=10, label="Steps",
                                                             visible=True, interactive=True)
                                scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                         label="Guidance Scale", interactive=True)
                                img_H_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                         label="Img Height", interactive=True)
                                img_W_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                         label="Img Width", interactive=True)
                            gr.Markdown(value=parameter_description)

                ex = gr.Examples(examples=examples,
                                 inputs=[prompt_box, ddim_step_slider, scale_slider, seed_box],
                                 outputs=[output_gallery, seed_box],
                                 fn=gr_interface_un_save,
                                 examples_per_page=15,
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

                prompt_box.submit(text2img_infer,
                                  inputs=[prompt_box, seed_box, ddim_step_slider, scale_slider, img_H_slider,
                                          img_W_slider, random_seed_checkbox],
                                  outputs=[output_gallery, seed_box])

                generate_button.click(text2img_infer,
                                      inputs=[prompt_box, seed_box, ddim_step_slider, scale_slider, img_H_slider,
                                              img_W_slider, random_seed_checkbox],
                                      outputs=[output_gallery, seed_box])

        with gr.Tab("Img to Img"):
            with gr.Column():
                # gr.Column()   垂直      | gr.ROW()  水平
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 提示词 - (请勿超过64个词)")
                            prompt_box = gr.Textbox(label="prompts", lines=1, show_label=False)
                            generate_button = gr.Button("开始绘画", elem_id="go_button")
                        gr.Markdown("""
                        #### [翻译器](https://www.deepl.com/translator)   [探索提示词](https://openart.ai/)
                        ---
                        """)
                        with gr.Group():
                            init_image = gr.Image(shape=(512, 512), image_mode="RGB", source="upload",
                                                  type="filepath", tool="select", label="Init image",
                                                  show_label=True, interactive=True, visible=True,
                                                  elem_id="init_image")
                            canvas_init_image = gr.Image(shape=(512, 512), image_mode="RGB", source="canvas",
                                                         type="filepath", tool="select", label="Init image",
                                                         show_label=True, interactive=True, visible=False,
                                                         elem_id="init_image")
                            switch_button = gr.Button(value="switch to canvas", show_label=False)

                    with gr.Column():
                        gr.Markdown("""### 高级设置""")
                        with gr.Group():
                            with gr.Row():
                                seed_box = gr.Number(label="Seed", value=np.random.randint(1, 2147483646),
                                                     interactive=False, elem_id="seed_box")
                                random_seed_checkbox = gr.Checkbox(label="Random Seed", value=True, interactive=True,
                                                                   elem_id="random_seed")
                                ddim_step_slider = gr.Slider(minimum=10, maximum=50, step=1, value=10, label="Steps",
                                                             visible=args.step_un_show, interactive=True)
                                scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                         label="Guidance Scale", interactive=True)
                                strength_slider = gr.Slider(minimum=0, maximum=0.9, step=0.1, value=0.8,
                                                            label="Strength", interactive=True)
                                img_H_slider = gr.Slider(minimum=256, maximum=512, step=64, value=512,
                                                         label="Img Height", interactive=True,
                                                         visible=args.show_img_HW)
                                img_W_slider = gr.Slider(minimum=256, maximum=512, step=64, value=512,
                                                         label="Img Width", interactive=True,
                                                         visible=args.show_img_HW)
                        gr.Markdown("""#### Result images""")
                        output_gallery = gr.Gallery(interactive=True, label="Result gallery", show_label=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(value=prompt_note)
                    with gr.Column():
                        gr.Markdown(value=parameter_description_img2img)

                gr.Markdown(value=end_message)

            # style
            prompt_box.style(rounded=(True, True, False, False), container=False)
            generate_button.style(margin=False, rounded=(False, False, True, True), full_width="True")
            switch_button.style(margin=False, rounded=(False, False, True, True), full_width="True")
            output_gallery.style(grid=2, height="1024px")

            # action
            # seed
            random_seed_checkbox.change(update_interactive,
                                        inputs=[random_seed_checkbox],
                                        outputs=[seed_box])

            # switch upload and canvas
            switch_button.click(update_visible, inputs=[switch_button],
                                outputs=[init_image, canvas_init_image, switch_button])

            # prompt
            prompt_box.submit(img2img_infer,
                              inputs=[prompt_box, init_image, canvas_init_image, seed_box, ddim_step_slider,
                                      strength_slider, scale_slider, img_H_slider, img_W_slider, random_seed_checkbox],
                              outputs=[output_gallery, seed_box])
            generate_button.click(img2img_infer,
                                  inputs=[prompt_box, init_image, canvas_init_image, seed_box, ddim_step_slider,
                                          strength_slider, scale_slider, img_H_slider, img_W_slider, random_seed_checkbox],
                                  outputs=[output_gallery, seed_box])

    advanced_app.queue(concurrency_count=2, max_size=15)
    advanced_app.launch(show_error=False, server_port=6006, share=False, quiet=False)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser("Text2img && Img2img")
    parser.add_argument("--out_dir", "-o", type=str, help="result output folder", nargs='?',
                        default="/root/Image_synthesis_webpage/stable-diffusion/outputs/")
    parser.add_argument("--ckpt", "-m", type=str, help="Stable-diffusion model checkpoint path",
                        default="/root/Image_synthesis_webpage/stable-diffusion/models/v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--config", "-c", type=str, help="Stable-diffusion model config path",
                        default="/root/Image_synthesis_webpage/stable-diffusion/configs/stable-diffusion/v2-inference.yaml")
    parser.add_argument("--step_un_show", "-u", action="store_false", help="whether step option is visible")
    parser.add_argument("--show_img_HW", "-s", action="store_true", help="show img width and img height slider")
    args = parser.parse_args()

    # profanity_filter
    profanity_filter = pron_filter("/root/Image_synthesis_webpage/stable-diffusion/utils/pron_blacklist.txt")
    draw_warning_img = Image.open("/root/Image_synthesis_webpage/stable-diffusion/utils/draw_warning.png")

    # load init_img warning
    init_warning_img = Image.open("/root/Image_synthesis_webpage/stable-diffusion/utils/add_init_img.png")

    # logging set
    log_set(show_level=logging.INFO, save_level=logging.INFO)

    # global save index
    global_index = 0

    # clear port
    clear_port(6006)
    gr.close_all()

    # load models
    all2img = all2img(ckpt=args.ckpt, config=args.config, output_dir=args.out_dir)
    logging.info("Successful initialization synthesis class")

    # run
    gr_advanced_vertical_page()