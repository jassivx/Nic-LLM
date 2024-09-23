import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your dataset from the JSON file
with open('data.json', 'r') as f:
    dataset = json.load(f)

# Load the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"  # Adjust the model based on your requirements
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Function to generate a response and measure response time
def generate_response(prompt, max_length=60):
    inputs = tokenizer(prompt, return_tensors='pt')

    start_time = time.time()  # Start timing
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.5,  # Lower temperature for more coherent responses
            top_k=50,
            top_p=0.9,  # Slightly higher top_p for diversity
            do_sample=True
        )
    end_time = time.time()  # End timing

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_time = end_time - start_time  # Calculate response time
    return response, response_time


# Example of using the loaded dataset
for example in dataset:
    prompt = example['prompt']
    expected_response = example['response']

    print(f"Prompt: {prompt}")
    print(f"Expected Response: {expected_response}")

    # Generate response from model
    response, response_time = generate_response(prompt)
    print(f"Model Response: {response}")
    print(f"Response Time: {response_time:.2f} seconds\n")

# Chat loop (for interaction)
print("Chat with the model (type 'exit' to stop):")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    response, response_time = generate_response(user_input)
    print(f"Model: {response}")
    print(f"Response Time: {response_time:.2f} seconds")
