from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from index import index
import gradio as gr
from admin import admin
from welcome import welcome
from display import get_show_page

app = FastAPI()
port = 7861

app.mount("/static", StaticFiles(directory="static"), name="static")

gr.mount_gradio_app(app, index, path="/index")
gr.mount_gradio_app(app, admin, path="/admin")
gr.mount_gradio_app(app, welcome, path="/welcome")


@app.get("/show_result/{demo_name}")
async def get_result(demo_name: str):
    gr.mount_gradio_app(app, get_show_page(demo_name),
                        path=f"/display/{demo_name}")
    return RedirectResponse(url=f"http://localhost:{port}/display/{demo_name}")


if __name__ == "__main__":
    print("System running...")
    uvicorn.run("main:app", port=port, reload=True,
                log_level="error", timeout_graceful_shutdown=3)
