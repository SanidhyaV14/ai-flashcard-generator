import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pydantic import BaseModel

class Flashcard(BaseModel):
    front: str
    back:str

class FlashcardSet(BaseModel):
    topic: str
    card: list[Flashcard]

def generate(text_input: str, num_cards: int):
    
    load_dotenv()
    client = InferenceClient(
        api_key=os.environ.get("HF_TOKEN"),
    )
    
    prompt = f"Create a set of {num_cards} flashcards based on the following text: {text_input}. Each flashcard should have a 'front' and a 'back'. The output should be in JSON format, structured as follows: {{'topic': '...', 'card': [{{'front': '...', 'back': '...'}}, ...]}}."

    resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        response_format={"type": "json_object", "value": FlashcardSet.model_json_schema()}
)

    return json.loads(resp.choices[0].message.content)


inp = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."

result = generate(inp, 4)

print(f"\nTopic: {result['topic']}")
for i, card in enumerate(result['card'], 1):
    print(f"Card {i} | Ques: {card['front']} -> Ans: {card['back']}")