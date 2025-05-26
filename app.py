import gradio as gr

from PIL import Image 
from PIL import ImageEnhance

#authorization token for Hugging Face API


from PIL import ImageEnhance

from huggingface_hub import InferenceClient


# Setup Inference Client with your API key and Nebius provider
client = InferenceClient(
    provider="nebius",
    api_key="add your Hugging Face API key here",
)

def generate_image(prompt, brightness=1.0, contrast=1.0):
    try:
        # Directly get a PIL.Image from the client
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev")

        # Apply brightness and contrast adjustments
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to generate image: {e}")


# Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=["text",gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Brightness"),
            gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Contrast")],
    outputs="image",
    live=True,
)

demo.launch(share=True)