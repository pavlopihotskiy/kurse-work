import random
from app.config import CLASSES


class TinyMLModel:
    def __init__(self):
        # заглушка, реальна модель не використовується
        pass

    def predict(self, features):
        predicted_class = random.choice(CLASSES)
        confidence = round(random.uniform(0.7, 0.95), 3)
        return predicted_class, confidence
