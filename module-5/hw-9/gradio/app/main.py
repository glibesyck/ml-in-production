import gradio as gr
from PIL import Image
from model import load_model, predict


def predict_handler(image):
    model = load_model()
    if image is None:
        return "Please upload an image"
    
    image_pil = Image.fromarray(image).convert("L").resize((224, 224))
    result = predict(model, image_pil)

    return "✅ Same figure!" if result else "❌ Different figures."

demo = gr.Interface(
    fn=predict_handler,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Text(),
    title="👀 Are the Points from the Same Figure?",
    description="Upload a 224x224 grayscale image to check.",
    examples=[],
)

if __name__ == "__main__":
    demo.launch(debug=True)
