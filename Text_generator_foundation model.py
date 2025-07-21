import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load AI detection model
MODEL_NAME = "roberta-base-openai-detector"  # Replace with your fine-tuned model if available

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load text generation pipeline (GPT-2)
text_generator = pipeline("text-generation", model="gpt2")

# Load image generation pipeline (Stable Diffusion)
image_generator = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
image_generator = image_generator.to("cuda" if torch.cuda.is_available() else "cpu")

# AI Detection Function
def detect_origin(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        ai_score = round(probs[0][1].item() * 100, 2)
        human_score = round(100 - ai_score, 2)
        return f"🧠 AI Likelihood: {ai_score}%\n👤 Human Likelihood: {human_score}%"

# Text Generation Function
def generate_text(prompt):
    result = text_generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

# Image Generation Function
def generate_image(prompt):
    image = image_generator(prompt=prompt).images[0]
    return image

# Gradio Interface
with gr.Blocks(title="AI Content Toolkit") as demo:
    gr.Markdown("# ✨ AI Content Toolkit")
    gr.Markdown("This app detects AI-generated text, generates new text, and creates images from prompts using powerful foundation models.")

    with gr.Tab("🧠 AI vs Human Text Detector"):
        text_input = gr.Textbox(lines=7, placeholder="Paste your text here...", label="Input Text")
        text_output = gr.Textbox(label="Result")
        detect_btn = gr.Button("Detect")
        detect_btn.click(detect_origin, inputs=text_input, outputs=text_output)

    with gr.Tab("📝 Text Generator"):
        prompt_input = gr.Textbox(lines=3, placeholder="Enter a prompt...", label="Text Prompt")
        generated_text = gr.Textbox(lines=10, label="Generated Text")
        textgen_btn = gr.Button("Generate Text")
        textgen_btn.click(generate_text, inputs=prompt_input, outputs=generated_text)

    with gr.Tab("🎨 Image Generator"):
        image_prompt = gr.Textbox(lines=2, placeholder="Enter a prompt for the image...", label="Image Prompt")
        image_output = gr.Image(type="pil", label="Generated Image")
        imgen_btn = gr.Button("Generate Image")
        imgen_btn.click(generate_image, inputs=image_prompt, outputs=image_output)

# Run the app
demo.launch()
