from fastapi import FastAPI
import uvicorn
from ui import index
import gradio as gr
from admin import admin
from display import display

app = FastAPI()

gr.mount_gradio_app(app, index, path="/index")
gr.mount_gradio_app(app, admin, path="/admin")
gr.mount_gradio_app(app, display, path="/display")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7861, reload=True, log_level="error")
    # demo.launch()
