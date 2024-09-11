import re
from typing import List
from scanner.scanner import Scanner

class BanWords(Scanner):
    """
    BanWords class is used to detect the presence of certain substrings in the prompt.
    """

    def __init__(self, words: List[str]) -> None:
        self._words = words

    def scan(self, prompt: str) -> tuple[str, float]:
        if any(re.search(rf"\b{re.escape(sub)}\b", prompt) for sub in self._words):
            return 'UNSAFE', 1.0
        return 'SAFE', -1.0
