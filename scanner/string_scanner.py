from typing import List
from scanner.scanner import Scanner

class BanSubstrings(Scanner):
    """
    BanSubstrings class is used to detect certain substrings in the prompt.
    """

    def __init__(self, substrings: List[str]) -> None:
        self._substrings = substrings

    def scan(self, prompt: str) -> tuple[str, float]:
        if any(substring in prompt for substring in self._substrings):
            return 'UNSAFE', 1.0
        return 'SAFE', 0.0