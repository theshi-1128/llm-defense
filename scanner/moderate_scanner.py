import torch
from llm.models import full_model_dict
from transformers import AutoTokenizer, AutoModelForCausalLM

class Moderator:
    def __init__(self, model_name: str, device: str, dtype=torch.bfloat16):
        """
        Initializes the Moderator class with the specified model and device.

        Parameters:
            model_name (str): The name of the model to use.
            device (str): The device on which the model will run (e.g., 'cuda:0').
            dtype (torch.dtype): The data type to use for the model. Default is torch.bfloat16.
        """
        self.model_name = model_name
        self.model_path = full_model_dict[self.model_name]["path"]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype, device_map=device)

    def moderate(self, chat, max_new_tokens=100, pad_token_id=0):
        """
        Moderates the given chat input using the model.

        Parameters:
            chat (list of dict): The input chat formatted as a list of dictionaries with 'role' and 'content' keys.
            max_new_tokens (int): The maximum number of tokens to generate. Default is 100.
            pad_token_id (int): The token ID used for padding. Default is 0.

        Returns:
            tuple: A tuple where the first element is 'SAFE' or 'UNSAFE' and the second element is a float score.
        """
        # Convert chat input to model input format
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)

        # Generate response using the model
        output = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id)

        # Extract the generated response
        prompt_len = input_ids.shape[-1]
        response = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        # Determine safety based on the response
        if 'safe' in response:
            return 'SAFE', 0
        else:
            return 'UNSAFE', 1.0
