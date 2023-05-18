import argparse
import json
import logging
import os
import shutil
import time
from uuid import uuid4

import gradio as gr
import numpy as np
from PIL import Image

from utils.all2img import all2img
from utils.logging_utils import log_set
from utils.prompt_note import parameter_description_img2img, parameter_description
from utils.prompt_note import prompt_note, end_message, examples
from utils.pron_filter import blacklist_filter as ProfanityFilter
from utils.YoudaoTranslate import YoudaoTranslate
from utils.utils import concat_img, clear_port


# build save path and save images(grid_image, init_image, samples)
def save_img(images: list, prompt: str, seed: int, ddim_steps: int, scale: float, strength: float = None,
             init_img_path: str = None, img_H=512, img_W=512, n_samples=4, n_iter=4, ddim_eta: float = 1.0,
             output_path="outputs"):
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
    config_dict = {"prompts": prompt, "seed": seed, "ddim_steps": ddim_steps, "ddim_eta": ddim_eta, "scale": scale,
                   "img_H": img_H, "img_W": img_W, "n_samples": n_samples, "n_iter": n_iter,
                   "grid_path": os.path.join(output_dir, "grid.png"), "samples_path": list()}

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
    filter_ = ProfanityFilter()
    filter_.add_from_file(blacklist_path)
    return filter_


# update 'random seed' button interactive value
def update_interactive(advanced_page):
    return gr.update(interactive=not advanced_page)


# update 'Image source' button visible value, between 'upload' and 'canvas'
def update_visible(switch_button):
    if switch_button == "switch to upload":
        return gr.update(visible=True), gr.update(visible=False), gr.update(value="switch to canvas")
    elif switch_button == "switch to canvas":
        return gr.update(visible=False), gr.update(visible=True), gr.update(value="switch to upload")


def update_state_value(evt: gr.SelectData):
    # logging.info(f"Switch to '{evt.value}' tab")
    return evt.value


def gr_interface_un_save(prompt, ddim_steps=50, scale=7.5, seed=1024, img_H=512, img_W=512, n_samples=4, n_iter=1,
                         ddim_eta=0.0):

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={ddim_steps}, scale={scale}, img_H={img_H}, img_W={img_W}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}, ddim_eta={ddim_eta}")

    all_samples = all2img.text2img(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter), img_H=img_H,
                                   img_W=img_W, ddim_steps=int(ddim_steps), scale=scale, ddim_eta=ddim_eta)
    images = all2img.postprocess(all_samples, single=True)

    return images, int(seed)


def text2img_infer(prompt, seed, ddim_steps, scale, img_H, img_W, random_seed, n_samples, n_iter, ddim_eta):
    global profanity_filter, draw_warning_img, args

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
    all_samples = all2img.text2img(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter), img_H=img_H,
                                   img_W=img_W, ddim_steps=int(ddim_steps), scale=scale, ddim_eta=ddim_eta)

    # postprocess
    images = all2img.postprocess(all_samples, single=True)

    # save
    save_img(images=images, prompt=prompt, seed=seed, ddim_steps=ddim_steps, ddim_eta=ddim_eta, scale=scale,
             img_H=img_H, img_W=img_W, n_samples=n_samples, n_iter=n_iter, output_path=args.out_dir)
    return images, int(seed)


# def img2img_infer():
def img2img_infer(prompt, init_img_path, canvas_init_path, seed, ddim_steps, strength, scale, img_H, img_W, random_seed,
                  n_samples, n_iter, ddim_eta):
    global profanity_filter, draw_warning_img, args, init_warning_img

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
             ddim_steps=ddim_steps, scale=scale, img_H=img_H, img_W=img_W, n_samples=n_samples, n_iter=n_iter,
             ddim_eta=ddim_eta, output_path=args.out_dir)
    return images, int(seed)


def generate_type(status, prompt: str, init_img_path: str, canvas_init_path: str,
                  text2img_seed=np.random.randint(1, 2147483646), img2img_seed=np.random.randint(1, 2147483646),
                  text2img_ddim_steps=50, img2img_ddim_steps=50, strength=0.8, text2img_scale=7.5, img2img_scale=7.5,
                  text2img_img_H=512, img2img_img_H=512, text2img_img_W=512, img2img_img_W=512,
                  text2img_random_seed=False, img2img_random_seed=False, n_samples=4, n_iter=1, ddim_eta=0.0):
    logging.info(f"Generate type: '{status}'")
    if status == "Text to Img":
        images, seed = text2img_infer(prompt, text2img_seed, text2img_ddim_steps, text2img_scale, text2img_img_H,
                                      text2img_img_W, text2img_random_seed, n_samples, n_iter, ddim_eta)
        return images, seed, None, img2img_seed
    elif status == "Img to Img":
        images, seed = img2img_infer(prompt, init_img_path, canvas_init_path, img2img_seed, img2img_ddim_steps,
                                     strength, img2img_scale, img2img_img_H, img2img_img_W, img2img_random_seed,
                                     n_samples, n_iter, ddim_eta)
        return None, text2img_seed, images, seed


def text_translate(text, target_language="英语"):
    if target_language == "英语":
        target_language = "en"
    elif target_language == "中文":
        target_language = "zh-CHS"
    translated = youdao_translate.text_translate(text, source_language="auto", target_language=target_language)
    logging.info(f"translate '{text}' to '{translated}'")
    return translated


def direct_translate_prompt(text):
    return gr.update(value=text_translate(text))


def send_target_to_prompt(translated_text):
    logging.info(f"Send '{translated_text}' to Prompt box")
    return gr.update(value=translated_text)


def gr_advanced_vertical_page():
    global args
    with gr.Blocks(title="AI_with_Art", css="utils/all2img.css") as demo:
        # prompt box
        with gr.Column():
            with gr.Row():
                gr.Markdown("### 提示词 - (请勿超过64个词)")
                auto_translate_button = gr.Button(value="Translate Prompt",
                                                  elem_classes="btn", elem_id="directTranslate")
            prompt_box = gr.Textbox(label="prompts", lines=1, placeholder="请输入提示词",
                                    show_label=False)
            generate_button = gr.Button("开始绘画", elem_classes="btn")

            prompt_box.style(container=False)
            generate_button.style(full_width=True)

            auto_translate_button.click(direct_translate_prompt, inputs=[prompt_box], outputs=[prompt_box])

        # text to img
        with gr.Tab("Text to Img", id=0) as text2img:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""#### Result images""")
                    text2img_output_gallery = gr.Gallery(interactive=False, label="Result gallery")

                with gr.Column():
                    gr.Markdown("### 高级设置")
                    with gr.Row():
                        text2img_seed_box = gr.Number(label="Seed", value=np.random.randint(1, 2147483646),
                                                      interactive=False)
                        text2img_random_seed_checkbox = gr.Checkbox(label="Random Seed", value=True)
                    with gr.Row():
                        text2img_ddim_step_slider = gr.Slider(minimum=10, maximum=50, step=1, value=10, label="Steps",
                                                              visible=args.step_un_show)
                        text2img_scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                          label="Guidance Scale")
                    with gr.Row():
                        text2img_img_H_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                          label="Img Height")
                        text2img_img_W_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                          label="Img Width")
                    gr.Markdown(value=parameter_description)

            ex = gr.Examples(examples=examples,
                             inputs=[prompt_box, text2img_ddim_step_slider, text2img_scale_slider, text2img_seed_box],
                             outputs=[text2img_output_gallery, text2img_seed_box], fn=gr_interface_un_save,
                             examples_per_page=8, cache_examples=args.un_cache_examples)
            ex.dataset.headers = [""]

            gr.Markdown(prompt_note)

            # style
            text2img_output_gallery.style(columns=2, height="auto")

            # action
            text2img_random_seed_checkbox.change(update_interactive, inputs=[text2img_random_seed_checkbox],
                                                 outputs=[text2img_seed_box])

        with gr.Tab("Img to Img", id=1) as img2img:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""#### Init image""")
                    init_image = gr.Image(shape=(512, 512), image_mode="RGB", source="upload", type="filepath",
                                          tool="select", label="Init image", show_label=True, visible=True)
                    canvas_init_image = gr.Image(shape=(512, 512), image_mode="RGB", source="canvas", type="filepath",
                                                 tool="select", label="Init image", show_label=True, visible=False)
                    switch_button = gr.Button(value="switch to canvas", show_label=False, elem_classes="btn")

                with gr.Column():
                    gr.Markdown("""#### Result images""")
                    img2img_output_gallery = gr.Gallery(label="Result gallery")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""### 高级设置""")
                    with gr.Row():
                        img2img_seed_box = gr.Number(label="Seed", value=np.random.randint(1, 2147483646),
                                                     interactive=False)
                        img2img_random_seed_checkbox = gr.Checkbox(label="Random Seed", value=True)
                        img2img_ddim_step_slider = gr.Slider(minimum=10, maximum=50, step=1, value=10, label="Steps",
                                                             visible=args.step_un_show)
                        img2img_scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                         label="Guidance Scale")
                        img2img_strength_slider = gr.Slider(minimum=0, maximum=0.9, step=0.1, value=0.8,
                                                            label="Strength")
                        img2img_img_H_slider = gr.Slider(minimum=256, maximum=512, step=64, value=512,
                                                         label="Img Height", visible=args.show_img_HW)
                        img2img_img_W_slider = gr.Slider(minimum=256, maximum=512, step=64, value=512,
                                                         label="Img Width", visible=args.show_img_HW)

            with gr.Row():
                with gr.Column():
                    gr.Markdown(value=prompt_note)
                with gr.Column():
                    gr.Markdown(value=parameter_description_img2img)

            # style
            switch_button.style(full_width=True)
            img2img_output_gallery.style(columns=2, height="auto")

            # action
            # seed
            img2img_random_seed_checkbox.change(update_interactive, inputs=[img2img_random_seed_checkbox],
                                                outputs=[img2img_seed_box])

            # switch upload and canvas
            switch_button.click(update_visible, inputs=[switch_button],
                                outputs=[init_image, canvas_init_image, switch_button])

        with gr.Tab("Prompt Translate", id=2) as prompt_translate:
            target_language = gr.Dropdown(choices=["中文", "英文"], value="英文", label="Target Language", type="value",
                                          multiselect=False, allow_custom_value=False, interactive=True)
            with gr.Row():
                with gr.Column():
                    source_text = gr.Textbox(label="Source Text", lines=3, max_lines=10, interactive=True)
                with gr.Column():
                    translated_text = gr.Textbox(label="Target Text", lines=3, max_lines=10, interactive=True)
            with gr.Row():
                translate_button = gr.Button(value="Translate", show_label=False, elem_classes="btn")
                send_button = gr.Button(value="Send to Prompt Box", show_label=False, elem_classes="btn")

            source_text.submit(text_translate, inputs=[source_text, target_language], outputs=[translated_text])
            translate_button.click(text_translate, inputs=[source_text, target_language], outputs=[translated_text])
            send_button.click(send_target_to_prompt, inputs=[translated_text], outputs=[prompt_box])

        gr.Markdown(value=end_message)

        # tab message state
        used_tab = gr.State("Text to Img")
        text2img.select(update_state_value, None, used_tab)
        img2img.select(update_state_value, None, used_tab)
        prompt_translate.select(update_state_value, None, used_tab)

        # generate action
        prompt_box.submit(generate_type,
                          inputs=[used_tab, prompt_box, init_image, canvas_init_image, text2img_seed_box,
                                  img2img_seed_box, text2img_ddim_step_slider, img2img_ddim_step_slider,
                                  img2img_strength_slider, text2img_scale_slider, img2img_scale_slider,
                                  text2img_img_H_slider, img2img_img_H_slider, text2img_img_W_slider,
                                  img2img_img_W_slider, text2img_random_seed_checkbox,
                                  img2img_random_seed_checkbox],
                          outputs=[text2img_output_gallery, text2img_seed_box, img2img_output_gallery,
                                   img2img_seed_box])

        generate_button.click(generate_type,
                              inputs=[used_tab, prompt_box, init_image, canvas_init_image, text2img_seed_box,
                                      img2img_seed_box, text2img_ddim_step_slider, img2img_ddim_step_slider,
                                      img2img_strength_slider, text2img_scale_slider, img2img_scale_slider,
                                      text2img_img_H_slider, img2img_img_H_slider, text2img_img_W_slider,
                                      img2img_img_W_slider, text2img_random_seed_checkbox,
                                      img2img_random_seed_checkbox],
                              outputs=[text2img_output_gallery, text2img_seed_box, img2img_output_gallery,
                                       img2img_seed_box])

    demo.queue(concurrency_count=2, max_size=15, status_update_rate="auto")
    demo.launch(show_error=False, share=False, quiet=False, server_port=6006)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser("Text2img && Img2img")
    parser.add_argument("--out_dir", "-o", type=str, help="result output folder", nargs='?', default="./outputs")
    parser.add_argument("--ckpt", "-m", type=str, help="Stable-diffusion model checkpoint path",
                        default="./models/v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--config", "-c", type=str, help="Stable-diffusion model config path",
                        default="./configs/stable-diffusion/v2-inference.yaml")
    parser.add_argument("--step_un_show", action="store_false", help="whether step option is visible")
    parser.add_argument("--un_cache_examples", action="store_true", help="Whether to cache examples")
    parser.add_argument("--show_img_HW", action="store_true", help="show img width and img height slider")
    args = parser.parse_args()

    # ------ 调试用参数 ------
    # args.out_dir = "/root/Image_synthesis_webpage/stable-diffusion/outputs/"
    # args.ckpt = "/root/Image_synthesis_webpage/stable-diffusion/models/realisticVisionV20_v20.safetensors"
    # args.config = "/root/Image_synthesis_webpage/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    # args.un_cache_examples = False
    # ------ 调试用参数 ------

    # profanity_filter
    profanity_filter = pron_filter("/root/Image_synthesis_webpage/stable-diffusion/utils/pron_blacklist.txt")
    draw_warning_img = Image.open("/root/Image_synthesis_webpage/stable-diffusion/utils/draw_warning.png")

    # load init_img warning
    init_warning_img = Image.open("/root/Image_synthesis_webpage/stable-diffusion/utils/add_init_img.png")

    # logging set
    log_set(log_level=logging.INFO, log_save=True)

    # global save index
    global_index = 0

    # clear port
    gr.close_all()
    clear_port(6006)

    # load models
    all2img = all2img(ckpt=args.ckpt, config=args.config, output_dir=args.out_dir)
    logging.info("Successful initialization synthesis class")

    # load translate tool
    # you can get this from https://ai.youdao.com/product-fanyi-text.s
    APP_KEY = 'YOUR APP KEY'
    APP_SECRET = 'YOUR APP SECRET'
    youdao_translate = YoudaoTranslate(APP_KEY=APP_KEY, APP_SECRET=APP_SECRET)

    # run
    gr_advanced_vertical_page()
