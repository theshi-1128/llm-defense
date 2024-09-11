from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class DefenseConfig:
    defense_method: str = field(default='None')
    no_system_prompt: bool = field(default=False)
    verbose: bool = field(default=False)

    def load_from_args(self, args: Any) -> None:
        """
        Load configuration settings from command-line arguments.

        Parameters:
            args: Command-line arguments containing configuration options.
        """
        self.no_system_prompt = args.no_system_prompt
        self.verbose = args.verbose

class DefenseBase:
    def __init__(self, config: DefenseConfig, preloaded_model: Optional[Any] = None, **kwargs: Any) -> None:
        """
        Initialize the base class for a defense method.

        Parameters:
            config (DefenseConfig): The defense configuration.
            preloaded_model (Optional[Any]): Optional preloaded model for the defense method.
            **kwargs: Additional keyword arguments.
        """
        self.defense_method = config.defense_method
        self.verbose = config.verbose
        self.preloaded_model = preloaded_model

    def defense(self, prompt: str, target_lm: Any, response: Optional[str] = None) -> str:
        """
        Apply the defense method to the given prompt.

        Parameters:
            prompt (str): The input prompt to be defended.
            target_lm (Any): The target language model to get the response.
            response (Optional[str]): An optional pre-existing response to use.

        Returns:
            str: The response after applying the defense method.
        """
        if response is None:
            response = target_lm.get_response([prompt], verbose=self.verbose)[0]
        return response
