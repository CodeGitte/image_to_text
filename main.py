
# Importing all libraries
from transformers import pipeline
from PIL import Image
import requests

# Defining the model pipeline from HuggingFace
classifier = pipeline(
    model="patrickjohncyh/fashion-clip", task="zero-shot-image-classification"
)

# Loading the images
url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
print(image)

# Article labels in German
fashion_articles = [
    "Kleid",
    "Hose",
    "Rock",
    "Bluse",
    "T-Shirt",
    "Pullover",
    "Jacke",
    "Mantel",
    "Anzug",
    "Schuhe",
    "Stiefel",
    "Sandalen",
    "Handtasche",
    "Gürtel",
    "Schal",
    "Hut",
    "Ohrringe",
    "Halskette",
    "Armband",
    "Ring",
]

# Prediction the labels by using the predefined classifer
predictions = classifier(image, candidate_labels=fashion_articles, multi_class=True)
print(predictions)
