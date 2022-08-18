import gradio as gr
import numpy as np
from text2img import make_args, text2img


def gr_interface(prompt, seed=None):
    global txt2img
    if seed is None:
        seed = np.random.randint(1, 100000)
    all_samples = txt2img.synthesis(prompt, seed=seed, n_samples=4, n_iter=1)
    return txt2img.save_img(all_samples, single_save=True, grid_save=True)[0], seed


def gr_main():
    # widgets
    with gr.Blocks() as demo:
        with gr.Column():
            with gr.Row():
                prompt = gr.Textbox(label="prompt", lines=1, max_lines=1)
                go_button = gr.Button("Go!")
                go_button.style(full_width=True, rounded=True)
            # with gr.Column():
            output_img = gr.Image()
            seed_box = gr.Number(interactive=False, label="seed")
            seed_box.style(rounded=True)

        # interface
        go_button.click(gr_interface, inputs=[prompt], outputs=[output_img, seed_box])

    # launch
    demo.launch(server_port=6006, share=False, quiet=False, enable_queue=True)


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
    txt2img = text2img(ckpt=opt.ckpt, config=opt.config, output_dir=opt.out_dir, n_rows=2)

    # main_class(opt)
    gr_main()
