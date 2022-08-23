import logging
import os
import json

import gradio as gr
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
from PIL import Image
from glob import glob
from uuid import uuid4
from min_dalle import MinDalle


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


def concat_img(images, grid_size=3, img_H=256, img_W=256):
    grid_img = Image.new("RGB", (grid_size * img_W, grid_size * img_H))
    for row in range(grid_size):
        for col in range(grid_size):
            grid_img.paste(images[grid_size * row + col], (0 + img_W * col, 0 + img_H * row))
    return grid_img


def post_process(images):
    return [Image.fromarray((img * 1).astype(np.uint8)).convert('RGB') for img in images.to('cpu').numpy()]


def stage_1(prompt, temperature=1, top_k="256", supercondition_factor="32", is_seamless=False):
    seed_list = [np.random.randint(0, 2 ** 32 - 1) for _ in range(4)]
    project_UUID = str(uuid4())
    logging.info(f"Project UUID: {project_UUID}")
    images = []
    with torch.no_grad():
        for seed in seed_list:
            images.append(post_process(model.generate_images(
                text=str(prompt),
                seed=seed,
                grid_size=1,
                is_seamless=is_seamless,
                temperature=int(temperature),
                top_k=int(top_k),
                supercondition_factor=int(supercondition_factor),
                is_verbose=False
            ))[0])

    stage_1_save_path = os.path.join("outputs", project_UUID, "stage_1")
    os.makedirs(stage_1_save_path, exist_ok=True)

    msg_dict = {"text": prompt,
                "grid_size": 1,
                "is_seamless": is_seamless,
                "temperature": int(temperature),
                "top_k-k": int(top_k),
                "supercondition_factor": int(supercondition_factor),
                "is_verbose": False,
                "seed_img": []}

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Grid_Size: {2}, Seamless: {is_seamless}, "
                 f"Temperature: {temperature}, Top_k: {top_k}, supercondition_factor: {supercondition_factor}")

    for seed, img in zip(seed_list, images):
        img_path = os.path.join(stage_1_save_path, f"{seed}.png")
        img.save(img_path)
        msg_dict["seed_img"].append({"seed": seed, "img_path": img_path})
        logging.info(f"seed: {seed}, img_path: {img_path}")
    with open(os.path.join(stage_1_save_path, "config.json"), "w") as json_f:
        json.dump(msg_dict, fp=json_f, ensure_ascii=False, sort_keys=True, indent=4, separators=(",", ": "))
    return images, seed_list[0], seed_list[1], seed_list[2], seed_list[3], project_UUID, gr.update(visible=True)


def stage_2(prompt, seed, project_UUID, temperature=1, top_k="128", supercondition_factor="32", is_seamless=False):
    with torch.no_grad():
        images = post_process(model.generate_images(
            text=str(prompt),
            seed=seed,
            grid_size=2,
            is_seamless=is_seamless,
            temperature=int(temperature),
            top_k=int(top_k),
            supercondition_factor=int(supercondition_factor),
            is_verbose=False
        ))

    stage_2_save_path = os.path.join("outputs", project_UUID, "stage_2")
    os.makedirs(stage_2_save_path, exist_ok=True)

    msg_dict = {"text": prompt,
                "seed": seed,
                "grid_size": 2,
                "is_seamless": is_seamless,
                "temperature": int(temperature),
                "top_k-k": int(top_k),
                "supercondition_factor": int(supercondition_factor),
                "is_verbose": False,
                "img_path": []}

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Grid_Size: {2}, seed: {seed}, Seamless: {is_seamless}, "
                 f"Temperature: {temperature}, Top_k: {top_k}, supercondition_factor: {supercondition_factor}")

    basic_i = len(glob(os.path.join(stage_2_save_path, "/*.png")))
    for index, img in enumerate(images):
        img_path = os.path.join(stage_2_save_path, f"{(index + basic_i):05}.png")
        logging.info(f"save img: {img_path}")
        img.save(img_path)
        msg_dict["img_path"].append(img_path)
    with open(os.path.join(stage_2_save_path, "config.json"), "w") as json_f:
        json.dump(msg_dict, fp=json_f, ensure_ascii=False, sort_keys=True, indent=4, separators=(",", ": "))
    return images


def gradio_2stage_app():
    with gr.Blocks(title="109美术高中AI与美术融合课", css="utils/gradio_css.css") as advanced_app:
        with gr.Column():
            gr.Markdown("## 109美术高中AI与美术融合课")
            with gr.Row():
                # 左半边
                with gr.Column():
                    with gr.Group():
                        prompt_box = gr.Textbox(label="提示词", lines=1)
                        go_button = gr.Button("开始绘画", elem_id="go_button")
                    output_gallery = gr.Gallery(interactive=False, show_label=False, elem_id="output_gallery")

                # 右半边
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### Choices Stage 2 IMG")
                        with gr.Row(visible=False) as stage_2_panel:
                            with gr.Column():
                                with gr.Group():
                                    U1_seed = gr.Number(show_label=False, value=np.random.randint(0, 2 ** 32 - 1),
                                                        visible=True, interactive=False)
                                    U1_button = gr.Button("Use IMG 1")
                            with gr.Column():
                                with gr.Group():
                                    U2_seed = gr.Number(show_label=False, value=np.random.randint(0, 2 ** 32 - 1),
                                                        visible=True, interactive=False)
                                    U2_button = gr.Button("Use IMG 2")
                            with gr.Column():
                                with gr.Group():
                                    U3_seed = gr.Number(show_label=False, value=np.random.randint(0, 2 ** 32 - 1),
                                                        visible=True, interactive=False)
                                    U3_button = gr.Button("Use IMG 3")
                            with gr.Column():
                                with gr.Group():
                                    U4_seed = gr.Number(show_label=False, value=np.random.randint(0, 2 ** 32 - 1),
                                                        visible=True, interactive=False)
                                    U4_button = gr.Button("Use IMG 4")

                    gr.Markdown("### Advanced Setting")
                    with gr.Row():
                        temperature_slider = gr.Slider(value=1, minimum=1, maximum=7, step=0.1, label="Temperature")
                        top_k_dropdown = gr.Dropdown(label="Top-k", value="128",
                                                     choices=[str(2 ** i) for i in range(15)], interactive=True)
                        supercondition_dropdown = gr.Dropdown(label="Super Condition", value="32",
                                                              choices=[str(2 ** i) for i in range(2, 7)],
                                                              interactive=True)
                        seamless_checkbox = gr.Checkbox(value=False, label="Seamless")
                    gr.Markdown(
                        """
                        ####
                        - **Input Text**: For long prompts, only the first 64 text tokens will be used to generate the image.
                        - **Seamless**: Tile images in image token space instead of pixel space.
                        - **Temperature**: High temperature increases the probability of sampling low scoring image tokens.
                        - **Top-k**: Each image token is sampled from the top-k scoring tokens.
                        - **Super Condition**: Higher values can result in better agreement with the text.
                        """
                    )

                    # invisible widgets
                    project_uuid = gr.Textbox(value="", visible=True, show_label=True, label="Project UUID")

            gr.Examples(examples=[
                ["cat, sticker, illustration, japanese style"],
                ["cyberpunk city"],
                ["A magnificent picture never seen before"],
                ["what it sees as very very beautiful"],
                ["a new creature"],
                ["What does the legendary phoenix  look like"],
                ["Pug hedgehog hybrid"],
                ["photo realistic, 4K, ultra high definition, cinematic, sea dragon horse"],
                ["city of coca cola oil painting"],
                ["dream come true"],
                ["AI robot teacher and students kid in classroom "],
                [
                    "wasteland, space station, cyberpunk, giant ship, photo realistic, 8K, ultra high definition, cinematic"],
                [
                    "sunset, sunrays showing through the woods in front, clear night sky, stars visible, mountain in the back, lake in front reflecting the night sky and mountain, photo realistic, 8K, ultra high definition, cinematic"],
            ], inputs=[prompt_box], examples_per_page=40)

        # style
        go_button.style(full_width="True", rounded=True)
        output_gallery.style(grid=[2], height="auto")
        U1_button.style(full_width="True", rounded=True)
        U2_button.style(full_width="True", rounded=True)
        U3_button.style(full_width="True", rounded=True)
        U4_button.style(full_width="True", rounded=True)

        # stage 1 action
        prompt_box.submit(stage_1,
                          inputs=[prompt_box, temperature_slider, top_k_dropdown,
                                  supercondition_dropdown, seamless_checkbox],
                          outputs=[output_gallery, U1_seed, U2_seed, U3_seed, U4_seed, project_uuid, stage_2_panel])
        go_button.click(stage_1,
                        inputs=[prompt_box, temperature_slider, top_k_dropdown,
                                supercondition_dropdown, seamless_checkbox],
                        outputs=[output_gallery, U1_seed, U2_seed, U3_seed, U4_seed, project_uuid, stage_2_panel])

        # stage 2 action
        U1_button.click(stage_2,
                        inputs=[prompt_box, U1_seed, project_uuid,
                                temperature_slider, top_k_dropdown, supercondition_dropdown, seamless_checkbox],
                        outputs=[output_gallery])
        U2_button.click(stage_2,
                        inputs=[prompt_box, U2_seed, project_uuid,
                                temperature_slider, top_k_dropdown, supercondition_dropdown, seamless_checkbox],
                        outputs=[output_gallery])
        U3_button.click(stage_2,
                        inputs=[prompt_box, U3_seed, project_uuid,
                                temperature_slider, top_k_dropdown, supercondition_dropdown, seamless_checkbox],
                        outputs=[output_gallery])
        U4_button.click(stage_2,
                        inputs=[prompt_box, U4_seed, project_uuid,
                                temperature_slider, top_k_dropdown, supercondition_dropdown, seamless_checkbox],
                        outputs=[output_gallery])

    advanced_app.launch(server_port=6006, share=False, quiet=False, enable_queue=True, show_error=True)


if __name__ == '__main__':
    log_set()

    # load model
    logging.info("Start load min-dalle model")
    model = MinDalle(models_root='./models', dtype=torch.float32,
                     device='cuda', is_mega=True, is_reusable=True)
    logging.info("Successful load min-dalle models")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # webpage
    # gradio_basic_page()
    gradio_2stage_app()
