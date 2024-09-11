from typing import Protocol

class Scanner(Protocol):
    """
    Scanner protocol that defines the interface for scanners.
    """

    def scan(self, prompt: str) -> tuple[str, float]:
        """
        Process and evaluate the input prompt according to the scanner's implementation.

        Parameters:
            prompt (str): The input prompt that needs to be evaluated.

        Returns:
            tuple[str, float]: A tuple where:
                - str: A flag indicating whether the prompt is valid ('SAFE') or not ('UNSAFE').
                - float: A risk score where 0.0 means no risk and 1.0 means high risk.
        """
