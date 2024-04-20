import torch
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

######## model here.................
model_path = './checkpoints/llava-v1.5-mix665k-NWPU-RSICD-UAV-LR-DOTA-FineTuneChat-lora-Bazi-neural-chat-02Feb'
model_base = 'Intel/neural-chat-7b-v3-3'

#### Further instrcutions here..........
conv_mode = 'llava_v1'
disable_torch_init()
model_path = os.path.abspath(model_path)
print('model path')
print(model_path)
model_name = get_model_name_from_path(model_path)
print('model name')
print(model_name)
print('model base')
print(model_base)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)


def chat_with_RS_LLaVA(cur_prompt,image_name):
    # Prepare the input text, adding image-related tokens if needed
    image_mem = Image.open(image_name)
    image_tensor = image_processor.preprocess(image_mem, return_tensors='pt')['pixel_values'][0]

    if model.config.mm_use_im_start_end:
        cur_prompt = f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}\n{cur_prompt}"
    else:
        cur_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{cur_prompt}"

    # Create a copy of the conversation template
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], cur_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Process image inputs if provided
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) .cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            no_repeat_ngram_size=3,
            max_new_tokens=2048,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    return outputs




if __name__ == "__main__":


    print('Model input...............')
    cur_prompt='Generate three questions and answers about the content of this image. Then, compile a summary.'
    image_name='./ex_img/parking_lot_010.jpg'

    outputs=chat_with_RS_LLaVA(cur_prompt,image_name)
    print('Model Response.....')
    print(outputs)
