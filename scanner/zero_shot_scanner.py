from scanner.scanner import Scanner
from transformers import pipeline
from llm.models import full_model_dict

class ZeroShot(Scanner):
    def __init__(self, topics: list[str] = ['unsafe', 'safe'], *, model_name: str, device: str) -> None:
        self._topics = topics
        self._classifier = pipeline(
            task="zero-shot-classification",
            model=full_model_dict[model_name]["path"],
            device=device
        )

    def scan(self, prompt: str) -> tuple[str, float]:
        output = self._classifier(prompt, self._topics, hypothesis_template="This text is about {}")
        scores = output['scores']
        labels = output['labels']
        max_index = scores.index(max(scores))
        return labels[max_index].upper(), scores[max_index]
