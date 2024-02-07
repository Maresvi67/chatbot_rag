from chatdoc.debug import FakeChatModel
from langchain.chat_models.base import BaseChatModel
from huggingface_hub import InferenceClient

def get_llm(model: str, **kwargs) -> BaseChatModel:
    if model == "debug":
        return FakeChatModel()

    if "mistral" in model:
        return InferenceClient(model)

    raise NotImplementedError(f"Model {model} not supported!")

# prompt__ = "This is a conversation, Always generate grammatically correct sentences that no repeat. The user is a very understanding. Respond to recent discussion between user and you"


# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", padding_side="left")
# tokenizer.pad_token = self.tokenizer.eos_token
# templ = tokenizer.apply_chat_template(msgs_list, tokenize=False)
# print(templ)

# llm_output = client.text_generation(prompt__ + templ, max_new_tokens=512)    
