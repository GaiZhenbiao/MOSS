import os

import subprocess
# Use all free GPUs
def get_available_gpu_ids():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                        '--format=csv,noheader,nounits'])
    gpu_list = output.decode('utf-8').strip().split('\n')
    available_gpus = []
    for gpu in gpu_list:
        index, memory_used = map(int, gpu.split(','))
        if memory_used <= 100:
            available_gpus.append(str(index))
    return ','.join(available_gpus)
GPUs = get_available_gpu_ids()
print("Available GPUs: {}".format(GPUs))
os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

import torch

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.utils import logger
from huggingface_hub import snapshot_download
import mdtex2html
import gradio as gr
import platform
import warnings
from moss_inference_streaming import Inference


try:
    from transformers import MossForCausalLM, MossTokenizer
except (ImportError, ModuleNotFoundError):
    from models.modeling_moss import MossForCausalLM
    from models.tokenization_moss import MossTokenizer
    from models.configuration_moss import MossConfig

infer = Inference(model_dir="fnlp/moss-moon-003-sft", device_map="auto")

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    query = parse_text(input)
    chatbot.append((query, ""))
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += '<|Human|>:' + old_query + '<eoh>'+ '<|MOSS|>:' + response + '<eom>'
    prompt += '<|Human|>: ' + input + '<eoh>'
    it = infer(prompt)
    for response in it:
        chatbot[-1] = (query, parse_text(response.lstrip("<|MOSS|>:")))
        history = history + [(query, response)]
        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">欢迎使用 MOSS 人工智能助手！</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
