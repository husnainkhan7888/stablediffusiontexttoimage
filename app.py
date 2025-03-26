import gradio as gr
from generate import generate_image

def generate(prompt):
    image = generate_image(prompt)
    return image

iface = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Text-to-Image AI",
    description="Generate images from text using AI."
)

iface.launch()
