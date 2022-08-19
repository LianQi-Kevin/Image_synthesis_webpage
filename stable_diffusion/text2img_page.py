import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


def infer(prompt):
    global num_samples
    with autocast("cuda"):
        images = pipe([prompt] * num_samples, guidance_scale=8)["sample"]
    return images


def gradio_basic_page():
    with gr.Blocks(css=".container { max-width: 800px; margin: auto; }") as demo:
        gr.Markdown("<h1><center>Stable Diffusion</center></h1>")
        gr.Markdown("Stable Diffusion is an AI model that generates images from any prompt you give!")
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    text = gr.Textbox(
                        label="Enter your prompt", show_label=False, max_lines=1
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    btn = gr.Button("Run").style(
                        margin=False,
                        rounded=(False, True, True, False),
                    )
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto"
            )
            text.submit(infer, inputs=text, outputs=gallery)
            btn.click(infer, inputs=text, outputs=gallery)

        gr.Markdown(
            """___
       <p style='text-align: center'>
       Created by CompVis and Stability AI
       <br/>
       </p>"""
        )

    demo.launch(server_port=6006, share=False, quiet=False, enable_queue=True, show_error=True)


if __name__ == '__main__':
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    pipe = StableDiffusionPipeline.from_pretrained('stable_diffusion/models/stable-diffusion-v1-3-diffusers', scheduler=lms)

    num_samples = 2

    gradio_basic_page()