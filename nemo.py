from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16"

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(model.generate(**tok("hello", return_tensors="pt").to("cuda")))