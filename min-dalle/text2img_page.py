import logging
import os

import gradio as gr
import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
from PIL import Image
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


def save_img(images, grid_size=3, save_path="outputs", save_single=True, save_grid=True):
    sample_path = os.path.join(save_path, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # save samples
    if save_single:
        samples_basic_i = len(os.listdir(sample_path))
        for index, img in enumerate(images):
            img.save(os.path.join(sample_path, f"{(index + samples_basic_i):05}.png"))
            logging.info(f"Successful save {(index + samples_basic_i):05}.png")

    # save grid
    if save_grid:
        grid_basic_i = len(os.listdir(save_path)) - 1
        concat_img(images, grid_size=grid_size).save(os.path.join(save_path, f"grid-{grid_basic_i:04}.png"))
        logging.info(f"Successful save grid-{grid_basic_i:04}.png")
        return f"grid-{grid_basic_i:04}.png"


def update_random_seed(advanced_page):
    if advanced_page:
        return gr.update(interactive=False, value=-1)
    else:
        return gr.update(interactive=True, value=np.random.randint(1, 10000000))


def run_model(prompt, seed=-1, grid_size=2, is_seamless=False, temperature=1, top_k="256", supercondition_factor="32"):
    global model

    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    with torch.no_grad():
        images = model.generate_images(
            text=str(prompt),
            seed=seed,
            grid_size=grid_size,
            is_seamless=is_seamless,
            temperature=temperature,
            top_k=int(top_k),
            supercondition_factor=int(supercondition_factor),
            is_verbose=False
        )
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Seed: {seed}, Grid_Size: {grid_size}, Seamless: {is_seamless}, "
                 f"Temperature: {temperature}, Top_k: {top_k}, supercondition_factor: {supercondition_factor}")
    images = [Image.fromarray((img * 1).astype(np.uint8)).convert('RGB') for img in images.to('cpu').numpy()]
    # grid_img_path = save_img(images, grid_size, save_path="./outputs")
    grid_img_path = save_img(images, grid_size, save_path="./examples/")
    return images, grid_img_path


def gradio_basic_page():
    with gr.Blocks(title="109美术高中AI与美术融合课", css="utils/gradio_css.css") as basic_page:
        # gr.Column() 垂直排列
        with gr.Column():
            gr.Markdown("## 109美术高中AI与美术融合课")
            # gr.Row() 水平排列
            with gr.Row():
                with gr.Column():
                    prompt_box = gr.Textbox(label="提示词", lines=1)
                    go_button = gr.Button("开始绘画", elem_id="go_button")
                    output_img = gr.Gallery(interactive=False, show_label=False).style(grid=[5], height="640px")
        # style
        go_button.style(full_width="True")

        # action
        prompt_box.submit(run_model, inputs=[prompt_box], outputs=[output_img])
        go_button.click(run_model, inputs=[prompt_box], outputs=[output_img])

    basic_page.launch(server_port=6006, share=False, quiet=False, enable_queue=False, show_error=True)


def gradio_advanced_app():
    with gr.Blocks(title="109美术高中AI与美术融合课", css="utils/gradio_css.css") as advanced_app:
        with gr.Column():
            gr.Markdown("## 109美术高中AI与美术融合课 - min(DALL·E)")
            with gr.Row():
                with gr.Column():
                    prompt_box = gr.Textbox(label="提示词", lines=1)
                    go_button = gr.Button("开始绘画", elem_id="go_button").style(full_width="True")
                    output_gallery = gr.Gallery(interactive=False, show_label=False, elem_id="output_gallery")
                with gr.Column():
                    gr.Markdown("### Setting")
                    seed_box = gr.Number(value=-1, label="Seed",
                                         interactive=False, precision=0)
                    with gr.Row():
                        grid_size_slider = gr.Slider(value=2, minimum=1, maximum=5, step=1, label="Grid Size")
                        random_seed_checkbox = gr.Checkbox(value=True, label="Random Seed")
                        seamless_checkbox = gr.Checkbox(value=False, label="Seamless")
                    gr.Markdown("#### Advanced")
                    with gr.Row():
                        temperature_slider = gr.Slider(value=1, minimum=1, maximum=7, step=0.1, label="Temperature")
                        top_k_dropdown = gr.Dropdown(label="Top-k", value="256",
                                                     choices=[str(2 ** i) for i in range(15)], interactive=True)
                        supercondition_dropdown = gr.Dropdown(label="Super Condition", value="16",
                                                              choices=[str(2 ** i) for i in range(2, 7)],
                                                              interactive=True)
                    gr.Markdown(
                        """
                        ####
                        - **Input Text**: For long prompts, only the first 64 text tokens will be used to generate the image.
                        - **Seed**: Use a positive seed for reproducible results.
                        - **Random Seed**: If random seed, seed will set to -1
                        - **Grid Size**: Size of the image grid. 3x3 takes about 15 seconds.
                        - **Seamless**: Tile images in image token space instead of pixel space.
                        - **Temperature**: High temperature increases the probability of sampling low scoring image tokens.
                        - **Top-k**: Each image token is sampled from the top-k scoring tokens.
                        - **Super Condition**: Higher values can result in better agreement with the text.
                        """
                    )

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
                ["wasteland, space station, cyberpunk, giant ship, photo realistic, 8K, ultra high definition, cinematic"],
                ["sunset, sunrays showing through the woods in front, clear night sky, stars visible, mountain in the back, lake in front reflecting the night sky and mountain, photo realistic, 8K, ultra high definition, cinematic"],
            ], inputs=[prompt_box], examples_per_page=40)

        # style
        go_button.style(full_width="True", rounded=True)
        seed_box.style(rounded=True)
        output_gallery.style(grid=[3], height="auto")

        # action
        random_seed_checkbox.change(update_random_seed, inputs=[random_seed_checkbox], outputs=[seed_box])
        prompt_box.submit(run_model,
                          inputs=[prompt_box, seed_box, grid_size_slider, seamless_checkbox,
                                  temperature_slider, top_k_dropdown, supercondition_dropdown],
                          outputs=[output_gallery])
        go_button.click(run_model,
                        inputs=[prompt_box, seed_box, grid_size_slider, seamless_checkbox,
                                temperature_slider, top_k_dropdown, supercondition_dropdown],
                        outputs=[output_gallery])

    advanced_app.launch(server_port=6006, share=False, quiet=False, enable_queue=True, show_error=True)


if __name__ == '__main__':
    log_set()

    # load model
    logging.info("Start load min-dalle model")
    model = MinDalle(models_root='./models', dtype=torch.float32,
                     device='cuda', is_mega=True, is_reusable=True)
    logging.info("Successful load min-dalle models")

    # webpage
    # gradio_basic_page()
    gradio_advanced_app()
