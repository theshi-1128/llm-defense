from transformers import AutoModelForCausalLM, AutoTokenizer


full_model_dict = {
    "glm4": {
        "path": "/home/model/glm-4-9b-chat",
    },
    "llama-guard": {
        "path": "/home/model/llama_guard_3_8B",
    },
    "deberta": {
        "path": "/home/model/deberta-v3-base-prompt-injection-v2",
    },
    "zero-shot": {
        "path": "/home/model/deberta-v3-large-zeroshot-v2.0",
    },
}

class TargetLM(object):
    def __init__(
            self,
            model_name: str,
            device: str,
            dtype: str = "auto",
            max_n_tokens: int = 1024,
            temperature: float = 0.0,
            top_p: float = 1.0,
    ):
        self.dtype = dtype
        self.device = device
        self.model_name = model_name
        self.model_path = full_model_dict[self.model_name]["path"]
        self.max_n_tokens = max_n_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.conversation_history: List[dict] = []

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        return tokenizer, model

    def get_response(self, user_input, final_result=None):
        if final_result == 'UNSAFE':
            return "I'm sorry, but I cannot assist with that request."

        self.conversation_history.append({"role": "user", "content": user_input})

        text = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=self.max_n_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[0].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()

        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


class DefendedTargetLM:
    def __init__(self, target_model, defense):
        self.target_model = target_model
        self.defense = defense

    def get_response(self, prompt):
        response = None
        defensed_response = self.defense.defense(prompt, self.target_model, response=response)
        if "I'm sorry, but I cannot assist with that request." in defensed_response:
            return 'UNSAFE', 1.0
        else:
            return 'SAFE', 0