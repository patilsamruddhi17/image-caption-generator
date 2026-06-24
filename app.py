from flask import Flask, render_template, request
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    output_ids = model.generate(pixel_values, max_length=20, num_beams=5)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            caption = generate_caption(image_path)

    return render_template("index.html", caption=caption, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)