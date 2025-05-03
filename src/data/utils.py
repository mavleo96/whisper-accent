import re


def preprocess_labels(text):
    text = re.sub(r"<[^>]+>", "", text)  # removes tags
    text = text.lower()  # lowercase
    # TODO: check if contractions needs to be handled separately
    text = re.sub(r"[^\w\s]", "", text)  # removes punctuation
    text = re.sub(r"\s+", " ", text).strip()  # removes extra spaces
    return text


if __name__ == "__main__":
    text = "it's like what are you doing? <overlap>."
    print(f"input: {text}")
    print(f"output: {preprocess_labels(text)}")
