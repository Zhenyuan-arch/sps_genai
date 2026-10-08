import torch
from torchvision import transforms
from PIL import Image
from fastapi import UploadFile, File
from app.classifier_model import CIFAR10CNN
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

app = FastAPI()
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

classifier = CIFAR10CNN().to(device)

classifier.load_state_dict(
    torch.load(
        "cifar10_classifier.pth",
        map_location=device
    )
)

classifier.eval()

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

classifier_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
nlp = spacy.load("en_core_web_lg")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}

@app.get("/embedding/{word}")
def get_embedding(word: str):
    embedding = calculate_embedding(word)

    return {
        "word": word,
        "embedding": embedding.tolist()
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    image_tensor = classifier_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(image_tensor)

        _, predicted = torch.max(
            outputs,
            1
        )

    predicted_class = cifar10_classes[
        predicted.item()
    ]

    return {
        "predicted_class": predicted_class
    }