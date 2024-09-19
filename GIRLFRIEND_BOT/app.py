from flask import Flask, request, jsonify, render_template
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import json

app = Flask(__name__)

# Load BlenderBot model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Load fine-tuning data
with open('data/fine_tune_data.json', 'r') as file:
    fine_tune_data = json.load(file)


def get_response(user_input):
    # Fine-tuning responses based on the JSON data
    for entry in fine_tune_data:
        if entry['user_message'].lower() in user_input.lower():
            return entry['bot_response']

    inputs = tokenizer.encode(user_input, return_tensors='pt')
    reply_ids = model.generate(inputs, max_length=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
