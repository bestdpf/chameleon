from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
import torch
from PIL import Image

model_id = '../chameleon-7b'

processor = ChameleonProcessor.from_pretrained(model_id)
model = ChameleonForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)

prompt = 'add a read hat for the photo <image>'
img = Image.open('./test/liuyifei.png')

inputs = processor(images=[img], text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)

# print(f'inputs is {inputs}')
for k,v in inputs.items():
    print(f'input k {k} {v.shape}')

generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
ret = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(ret)

