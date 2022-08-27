import json
import logging
import os
import time
from uuid import uuid4

import gradio as gr
import numpy as np
from PIL import Image

from utils.prompt_note import prompt_note, examples, parameter_description
from utils.text2img import make_args, text2img


def log_set():
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler('log.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def concat_img(images, grid_size=2, img_H=512, img_W=512):
    grid_img = Image.new("RGB", (grid_size * img_W, grid_size * img_H))
    for row in range(grid_size):
        for col in range(grid_size):
            grid_img.paste(images[grid_size * row + col], (0 + img_W * col, 0 + img_H * row))
    return grid_img


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


def gr_interface(prompt, seed=np.random.randint(1, 2147483646), ddim_steps=50, scale=7.5, img_H=512, img_W=512,
                 random_seed=False, n_samples=4, n_iter=1, ddim_eta=0.0):
    global txt2img

    if random_seed:
        seed = np.random.randint(1, 2147483646)
    else:
        seed = int(seed)

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={ddim_steps}, scale={scale}, img_H={img_H}, img_W={img_W}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}, ddim_eta={ddim_eta}")

    all_samples = txt2img.synthesis(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter),
                                    img_H=img_H, img_W=img_W, ddim_steps=int(ddim_steps), scale=scale,
                                    ddim_eta=ddim_eta)
    images = txt2img.postprocess(all_samples, single=True)

    save_img(images, prompt, seed, ddim_steps, scale, img_H, img_W, n_samples, n_iter, ddim_eta, output_path="outputs/")
    return images, int(seed)


def gr_interface_un_save(prompt, ddim_steps=50, scale=7.5, seed=1024, img_H=512, img_W=512,
                         n_samples=4, n_iter=1, ddim_eta=0.0):
    global txt2img

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, ddim_steps={ddim_steps}, scale={scale}, img_H={img_H}, img_W={img_W}")
    logging.info(f"n_samples={n_samples}, n_iter={n_iter}, ddim_eta={ddim_eta}")

    all_samples = txt2img.synthesis(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter),
                                    img_H=img_H, img_W=img_W, ddim_steps=int(ddim_steps), scale=scale,
                                    ddim_eta=ddim_eta)
    images = txt2img.postprocess(all_samples, single=True)

    return images, int(seed)


def update_interactive(advanced_page):
    if advanced_page:
        return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


def gr_advanced_page():
    global adv_visible

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
                        gr.Markdown("""
                        [翻译器](https://www.deepl.com/translator)
                        - - -
                        """)

                    with gr.Column():
                        gr.Markdown("### 高级设置")
                        with gr.Group():
                            with gr.Row():
                                seed_box = gr.Number(label="Seed", value=np.random.randint(1, 2147483646), interactive=False,
                                                     elem_id="seed_box")
                                random_seed_checkbox = gr.Checkbox(label="Random Seed", value=True, interactive=True,
                                                                   elem_id="random_seed")
                            with gr.Row():
                                ddim_step_slider = gr.Slider(minimum=10, maximum=50, step=1, value=10, label="Steps",
                                                             interactive=True)
                                scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                         label="Guidance Scale", interactive=True)
                                img_H_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                         label="Img Height", interactive=True)
                                img_W_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                         label="Img Width", interactive=True)
                                # ddim_eta_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="ddim_eta", interactive=True)
                                # n_sample_slider = gr.Slider(minimum=1, maximum=5, step=1, value=4, label="n_sample", interactive=True)
                                # n_iter_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="n_iter", interactive=True)
                            gr.Markdown(value=parameter_description)

                with gr.Column():
                    output_gallery = gr.Gallery(interactive=False).style(grid=[2], height="auto")
            ex = gr.Examples(examples=examples,
                             inputs=[prompt_box, ddim_step_slider, scale_slider, seed_box],
                             outputs=[output_gallery, seed_box],
                             fn=gr_interface_un_save,
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

    advanced_app.launch(server_port=6006, share=False, quiet=False, show_error=False)
    advanced_app.queue()


def gr_advanced_vertical_page():
    global adv_visible

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
                        gr.Markdown("[翻译器](https://www.deepl.com/translator)")
                    output_gallery = gr.Gallery(interactive=False).style(grid=[2], height="auto")

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
                                                         interactive=True)
                            scale_slider = gr.Slider(minimum=0, maximum=50, step=0.1, value=7.5,
                                                     label="Guidance Scale", interactive=True)
                            img_H_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                     label="Img Height", interactive=True)
                            img_W_slider = gr.Slider(minimum=384, maximum=512, step=64, value=512,
                                                     label="Img Width", interactive=True)
                            # ddim_eta_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="ddim_eta", interactive=True)
                            # n_sample_slider = gr.Slider(minimum=1, maximum=5, step=1, value=4, label="n_sample", interactive=True)
                            # n_iter_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="n_iter", interactive=True)
                        gr.Markdown(value=parameter_description)

                    ex = gr.Examples(examples=examples,
                                     inputs=[prompt_box, ddim_step_slider, scale_slider, seed_box],
                                     outputs=[output_gallery, seed_box],
                                     fn=gr_interface_un_save,
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
    # advanced_app.queue(concurrency_count=1, status_update_rate="auto", )


if __name__ == '__main__':
    # args
    opt = make_args()
    log_set()
    global_index = 0

    # ----------
    # 调试用 覆盖args
    opt.config = "./configs/stable-diffusion/v1-inference.yaml"
    opt.ckpt = "./models/stable-diffusion-v1-4-original/sd-v1-4.ckpt"
    opt.out_dir = "outputs/"  # output dir
    # ----------

    # kill all old gradio wrap
    gr.close_all()

    # init text2img
    txt2img = text2img(ckpt=opt.ckpt, config=opt.config, output_dir=opt.out_dir)

    # gr_basic_page()
    # gr_advanced_page()
    gr_advanced_vertical_page()
