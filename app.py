from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Lazy model loading to reduce memory usage
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer.padding_side = "left"
    return tokenizer, modell

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    return get_chat_response(msg)

MAX_HISTORY = 50  # Keep only the last 50 tokens

def get_chat_response(text):
    tokenizer, model = load_model()
    with torch.no_grad():
        input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors="pt")
        if len(input_ids[0]) > MAX_HISTORY:
            input_ids = input_ids[:, -MAX_HISTORY:]  # Truncate history

        chat_history_ids = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

    # Clear memory
    del model, tokenizer, input_ids, chat_history_ids
    torch.cuda.empty_cache()
    return response

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
