from scanner.scanner import Scanner
from llm.models import full_model_dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class PromptInjectionScanner(Scanner):
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.model_path = full_model_dict[self.model_name]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.device = device

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            device=self.device  # Ensure the device is correctly set
        )

    def scan(self, prompt: str) -> tuple[str, float]:
        result = self.classifier(prompt)[0]
        label = result['label']
        score = result['score']

        valid = 'SAFE' if label == 'SAFE' else 'UNSAFE'
        risk = score if valid == 'UNSAFE' else 0.0

        return valid, risk
