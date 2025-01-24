from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Lazy model loading to reduce memory usage
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.padding_side = "left"
    return tokenizer, model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    return get_chat_response(msg)

def get_chat_response(text):
    tokenizer, model = load_model()
    with torch.no_grad():
        input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = model.generate(input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

    # Clear memory
    del model, tokenizer, input_ids, chat_history_ids
    torch.cuda.empty_cache()
    return response

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
