import gradio as gr


def load_html():
    with open("static/welcome.html", "r", encoding="utf-8") as f:
        content = f.read()
    return content


with gr.Blocks() as welcome:
    gr.HTML(value=load_html())
