import os
import io
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

app = Flask(__name__)

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize Stability API client
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-fZYDryE4Q5sbDAV5E3GA7aix5QDEby6eUQmDmyDB7Qr1svWi'
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
    )

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']

    try:
        answers = stability_api.generate(
            prompt=prompt,
            seed=1,
            steps=50,
            cfg_scale=8.0,
            width=1024,
            height=1024,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn("Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    img_filename = 'generated_image.png'
                    img_path = os.path.join('static', img_filename)
                    img.save(img_path)
                    return render_template('indexx.html', image=img_filename)
    except Exception as e:
        print(f"Error generating image: {e}")  # Debug: Print any errors that occur
        return "There was an error generating the image."

if __name__ == '__main__':
    app.run(debug=True)
