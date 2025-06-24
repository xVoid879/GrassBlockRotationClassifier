from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return "API is running!"

@app.route("/classify", methods=["POST"])
def classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_path = f"temp/{image_file.filename}"
    image_file.save(image_path)

    # Run your classifier (example)
    result = subprocess.run(["python3", "main.py", "--image", image_path], capture_output=True, text=True)

    return jsonify({"output": result.stdout})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
