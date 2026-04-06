from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from model import predict_generated
from explain import generate_reason
import tempfile

app = Flask(__name__)
CORS(app)

# ✅ ADD THIS
@app.route("/")
def home():
    return "Backend is running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.json.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        prediction = predict_generated(text)
        reason = generate_reason(text)

        return jsonify({
            "prediction": prediction,
            "reason": reason
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": "Server error"}), 500


@app.route("/download", methods=["POST"])
def download():
    content = request.json.get("content", "")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    temp.write(content.encode())
    temp.close()

    return send_file(temp.name, as_attachment=True, download_name="report.txt")


if __name__ == "__main__":
    app.run(debug=True)