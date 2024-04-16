from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
import openai
import random
import re
import spacy 
import nltk
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
import re
import requests
# Load the GPT-3 API key from the .env file
load_dotenv()
# nltk.download('punkt')
# Set the OpenAI API key
openai.api_key = os.getenv("GPT3_API_KEY")

app = Flask(__name__)
# modificat aici in gpt 3.5 turbo de la text davinci - deja incepe sa ofere titlu !! modificat contetn in message
import re

def generate_text(prompt, example_base_text, desired_length, temperature):
    # Include a system message without the title
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{example_base_text}\nNow, write a sci-fi blog post about the following topic:\n\n{prompt}\n\nThe blog post should follow the style and tone of the example provided and should be set in a science-fiction context. It should include a title, an introduction, 2-3 main points,don't mention ', blog post,main points, point1, point2' kind of words, just be sure that it includes them, and a conclusion. Make sure it's engaging, well-structured, and approximately {desired_length} words long.Be sure to have a very intuitive and short title from maximum 4 words and try not to use quotes in the title."},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=desired_length,  # Use the desired length provided by the user
        temperature=temperature,
    )

    generated_text = response.choices[0].message["content"].strip()
    sentences = re.split(r'(?<=[.!?])\s+', generated_text)

    # Extract the title from the first sentence without including "Title:" and stopping at signs or "Introduction"
    title = sentences[0].replace("Title:", "").split("Introduction")[0].strip()
    # Check if the title ends with a period and remove it
    if title.endswith('.'):
        title = title[:-1]

    if not sentences[-1].endswith(('.', '?', '!')):
        sentences.pop()

    return ' '.join(sentences[1:]), title

# Return the generated text without the title and the extracted title
def generate_dalle_image(prompt):
    api_base = os.getenv("AZURE_OAI_ENDPOINT")
    api_key = os.getenv("AZURE_OAI_KEY")
    api_version = '2024-02-15-preview'
    
    url = f"{api_base}openai/deployments/dalle3/images/generations?api-version={api_version}"
    headers= { "api-key": api_key, "Content-Type": "application/json" }
    body = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    
    response = requests.post(url, headers=headers, json=body)
    return response.json()['data'][0]['url']


#def generate_title(generated_text):
    # Split the generated text into sentences
    #sentences = re.split(r'(?<=[.!?])\s+', generated_text)

    # Select the first sentence as the title
    #title = sentences[0].strip()

    #return title

def get_concise_prompt(generated_text):
    # Perform text summarization or keyword extraction to get representative keywords/phrases
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(generated_text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]

    # Create a concise prompt based on the extracted keywords
    concise_prompt = " ".join(keywords)
    
    # Limit the concise prompt to 1000 tokens if it's too long
    if len(concise_prompt) > 1000:
        concise_prompt = concise_prompt[:1000]

    return concise_prompt

def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read().splitlines()

def random_line(lines):
    return random.choice(lines)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompts = read_file("prom.txt")
    base_texts = read_file("base_text.txt")
    custom_prompt = request.form.get("custom_prompt")
    predefined_prompt = request.form.get("predefined_prompt")
    desired_length = int(request.form.get("length"))
    temperature = request.form.get("temperature")
    temperature = float(temperature) if temperature is not None else 0.7
    random_base_text = random_line(base_texts)

    # Use custom prompt if provided, otherwise use the predefined prompt or a random one from the text file
    prompt = custom_prompt if custom_prompt else (predefined_prompt if predefined_prompt else random_line(prompts))

    generated_text,generated_title = generate_text(prompt, random_base_text, desired_length, temperature)
    
    
    concise_prompt = get_concise_prompt(generated_text)
    
    generated_image_url = generate_dalle_image(concise_prompt)

    return render_template('generated.html', generated_text=generated_text, generated_title=generated_title, generated_image_url=generated_image_url)

if __name__ == '__main__':
    app.run(debug=True)
