import gradio as gr
import paddlehub as hub


"""
def generate_image(
          text_prompts:str,
          style: Optional[str] = "油画",
          topk: Optional[int] = 6,
          output_dir: Optional[str] = './ernievilg_output')

# 参数
text_prompts(str): 输入的语句，描述想要生成的图像的内容。
style(Optional[str]): 生成图像的风格，当前支持'油画','水彩','粉笔画','卡通','儿童画','蜡笔画','探索无限'。
topk(Optional[int]): 保存前多少张图，最多保存6张。
output_dir(Optional[str]): 保存输出图像的目录，默认为"ernievilg_output"。
     
# 提示词参照
https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/text_to_image/ernie_vilg#%E5%9B%9B-prompt-%E6%8C%87%E5%8D%97     
"""


def generate_img(text, style):
    global module
    return module.generate_image(text_prompts=[str(text)], style=str(style), output_dir='./ernie_vilg_out/')


def ernie_vilg_app():
    # global model
    with gr.Blocks(title="109美术高中AI与美术融合课") as advanced_app:
        # gr.Column()   垂直      | gr.ROW()  水平
        with gr.Column():
            gr.Markdown("""## 109美术高中AI与美术融合课
                - - -
                """)
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 提示词 - (请勿超过64个词)")
                    prompt_box = gr.Textbox(label="prompts", lines=1, show_label=False, placeholder="戴着眼镜的猫，漂浮在宇宙中，高更风格")
                    generate_button = gr.Button("开始绘画", elem_id="go_button")
                    style_Dropdown = gr.Dropdown(choices=['油画', '水彩', '粉笔画', '卡通', '儿童画', '蜡笔画', '探索无限'],
                                                 value="油画", label="风格", show_label=True, interactive=True)
                gr.Markdown("[提示词参考](http://www.youpromptme.cn/#/you-prompt-me/)")
                output_gallery = gr.Gallery(interactive=False).style(grid=[2], height="auto")
        gr.Markdown(
            """
            #### 
            ---
            - Model: [PaddlePaddle/ernie_vilg](https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/text_to_image/ernie_vilg)
            - UI Design: [刘学恺](https://github.com/LianQi-Kevin)

            """
        )
    generate_button.style(full_width="True")
    generate_button.click(fn=generate_img, inputs=[prompt_box, style_Dropdown], outputs=[output_gallery])
    prompt_box.submit(fn=generate_img, inputs=[prompt_box, style_Dropdown], outputs=[output_gallery])

    advanced_app.launch(server_port=6006, share=False, quiet=False, show_error=False, enable_queue=True)
    # advanced_app.queue(concurrency_count=1)


if __name__ == '__main__':
    print("Start Load ernie_vilg")
    module = hub.Module(name="ernie_vilg")
    print("Successful Load ernie_vilg")
    ernie_vilg_app()