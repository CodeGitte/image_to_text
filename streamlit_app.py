# Importing relevant libraries
import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# streamlit_app.py


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if check_password():
    # Adding title
    st.title("Mockup: Labeling von Artikelfotos")
    st.markdown(
        "Dieses Mockup ist beispielhaft für das Labeln von Artikelfotos. Hier können Artikelarten, wie zum Beispiel 'Hose' oder 'Bluse', sowie deren Farbe, wie zum Beispiel 'grün' oder 'blau', bestimmt werden. Diese Label wurden im Vorfeld konfiguriert und beschränken sich für das Mockup auf eine kleinere Anzahl, heißt hier gibt es zum Beispiel keine Unterscheidung innerhalb der Kategorie 'Schuhe' in Sandalen oder Stiefel. Es können eigene Fotos hochgeladen oder beispielhafte Artikelfotos genutzt werden."
    )
    st.divider()

    @st.cache_resource()
    def load_model():
        # Defining the model pipeline from HuggingFace
        return pipeline(
            model="patrickjohncyh/fashion-clip", task="zero-shot-image-classification"
        )

    # Load the model
    classifier = load_model()

    # Example images
    image_options_folder = "images"
    image_options = os.listdir(image_options_folder)

    # Drag and drop menu for new images
    uploaded_image = st.file_uploader(
        "Lade hier ein eigenes Artikelfoto hoch:", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, width=400)
    else:
        # Selecting image
        option = st.selectbox(
            "Oder wähle hier ein beispielhaftes Artikelfoto aus:",
            ["hier auswählen..."] + image_options,
        )

        if option != "hier auswählen...":
            image_path = os.path.join(image_options_folder, option)
            image = Image.open(image_path)
            st.image(image, width=400)

    if "image" in locals():
        # Article labels in German
        fashion_articles = [
            "Kleid",
            "Hose",
            "Rock",
            "Bluse",
            "T-Shirt",
            "Pullover",
            "Jacke",
            "Socken",
            "Mantel",
            "Schuhe",
            "Handtasche",
            "Hut",
            "Unterwäsche",
        ]

        # Color labels in German
        color_labels = [
            "rot",
            "blau",
            "grün",
            "gelb",
            "schwarz",
            "weiß",
            "grau",
            "braun",
            "pink",
            "beige",
            "grau" "violett",
        ]

        # Predicting the labels using the predefined classifier
        predictions_articles = classifier(
            image, candidate_labels=fashion_articles, multi_class=True
        )

        # Predicting the color labels using the predefined classifier
        predictions_colors = classifier(
            image, candidate_labels=color_labels, multi_class=True
        )

        # Displaying the predictions
        st.divider()
        st.header("Label")
        # Loop for article labels
        for prediction in predictions_articles[:1]:
            article_label = prediction["label"]
            article_score = round(prediction["score"], 2)

        # Loop for color labels
        for prediction in predictions_colors[:1]:
            color_label = prediction["label"]
            color_score = round(prediction["score"], 2)

        # Displaying the combined prediction
        st.write(
            f"Das zugewiesene Artikellabel ist '{article_label}' mit einem Score von {article_score}, wobei das zugewiesene Farbenlabel '{color_label}' ist mit einem Score von {color_score}."
        )
        markdown_text = """
            Die Punktzahl (Score) in diesem Zusammenhang repräsentiert das Vertrauen oder die Wahrscheinlichkeit, die dem Modell für jedes Label zugewiesen wird. Sie gibt die Einschätzung des Modells an, wie wahrscheinlich es ist, dass das gegebene Bild zu einem bestimmten Label gehört. Der Score liegt zwischen 0 und 1, wobei ein höherer Score eine größere Zuversichtlichkeit des Modells bedeutet, dass das Bild diesem spezifischen Label entspricht. Zum Beispiel, wenn das Modell dem Label 'Bluse' für ein bestimmtes Bild einen Score von 0,85 zuweist, bedeutet dies, dass das Modell zu 85% sicher ist, dass das Bild eine Bluse enthält. Ebenso deutet ein Score von 0,92 für das Label 'Schuhe' auf ein höheres Vertrauensniveau von 92% hin, dass das Bild Schuhe darstellt.
            """
        st.markdown(markdown_text)
