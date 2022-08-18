import gradio as gr
import numpy as np
from text2img import make_args, text2img


def gr_interface(prompt, seed=np.random.randint(1, 100000), img_H=256, img_W=256,
                 n_samples=4, n_iter=1, ddim_steps=50, ddim_eta=0.0, n_rows=2, scale=5.0):
    global txt2img
    all_samples = txt2img.synthesis(prompt, seed=seed, n_samples=int(n_samples), n_iter=int(n_iter),
                                    img_H=img_H, img_W=img_W, ddim_steps=int(ddim_steps), scale=scale, ddim_eta=ddim_eta)
    return txt2img.save_img(all_samples, single_save=True, grid_save=True, n_rows=n_rows)[0]


def gr_basic_page():
    with gr.Blocks(title="109美术高中AI与美术融合课") as basic_app:
        with gr.Column():
            with gr.Row():
                prompt = gr.Textbox(label="提示词", lines=1, max_lines=3)
                go_button = gr.Button("开始绘画")
                go_button.style(rounded=True)
            seed_box = gr.Number(interactive=True, label="seed", value=np.random.randint(1, 10000000))
            seed_box.style(rounded=True)
        go_button.click(gr_interface, inputs=[prompt, seed_box], outputs=[gr.Image()])
    basic_app.launch(server_port=6006, share=False, quiet=False, enable_queue=True, show_error=True)


def gr_advanced_page():
    with gr.Blocks(title="109美术高中AI与美术融合课") as advanced_app:
        # widgets
        with gr.Column():
            gr.Markdown("## 109美术高中AI与美术融合课")
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="提示词", lines=3)
                    go_button = gr.Button("开始绘画")
                    output_img = gr.Image()
                with gr.Column():
                    seed_box = gr.Number(interactive=True, label="seed", value=np.random.randint(1, 10000000))
                    with gr.Row():
                        ddim_step_slider = gr.Slider(minimum=30, maximum=80, step=1, value=50, label="ddim_step", interactive=True)
                        ddim_sta_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="ddim_eta", interactive=True)
                    with gr.Row():
                        n_sample_slider = gr.Slider(minimum=1, maximum=5, step=1, value=4, label="n_sample", interactive=True)
                        n_iter_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="n_iter", interactive=True)
                    with gr.Row():
                        img_H_slider = gr.Slider(minimum=256, maximum=512, step=64, value=256, label="图像高度", interactive=True)
                        img_W_slider = gr.Slider(minimum=256, maximum=512, step=64, value=256, label="图像宽度", interactive=True)

        # style
        go_button.style(rounded=True, full_width=True)
        seed_box.style(rounded=True)

        # action
        go_button.click(gr_interface,
                        inputs=[prompt, seed_box,
                                img_H_slider, img_W_slider,
                                n_sample_slider, n_iter_slider,
                                ddim_step_slider, ddim_sta_slider],
                        outputs=[output_img])
    advanced_app.launch(server_port=6006, share=False, quiet=False, enable_queue=True, show_error=True)


if __name__ == '__main__':
    # args
    opt = make_args()

    # ----------
    # 调试用 覆盖args
    opt.prompt = "sunset, sun rays showing through the woods in front, clear night sky, stars visible, mountain in the back, lake in front reflecting the night sky and mountain, photo realistic, 8K, ultra high definition, cinematic"
    opt.config = "/root/latent-Diffusion-Models/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    opt.ckpt = "/root/latent-Diffusion-Models/models/ldm/text2img-large/model.ckpt"
    opt.ddim_eta = 0.0  # 0.0 corresponds to deterministic sampling
    opt.n_samples = 3  # batch size    # X
    opt.n_iter = 3  # Y
    opt.scale = 5.0  # https://benanne.github.io/2022/05/26/guidance.html
    opt.ddim_steps = 50  # ddim sampling steps
    opt.out_dir = "outputs/txt2img-samples"  # output dir
    opt.H = 256  # Must be divisible by 8
    opt.W = 256  # Must be divisible by 8
    opt.skip_save = True  # skip save single img
    opt.skip_grid = False  # skip save grid img
    opt.seed = np.random.randint(1, 100)  # The same results when the seeds are the same
    # ----------

    # kill all old gradio wrap
    gr.close_all()

    # init text2img
    txt2img = text2img(ckpt=opt.ckpt, config=opt.config, output_dir=opt.out_dir)

    # gr_basic_page()
    gr_advanced_page()