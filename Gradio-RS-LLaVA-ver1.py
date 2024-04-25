import gradio as gr
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
model_path = 'BigData-KSU/RS-llava-v1.5-7b-LoRA'

model_base = 'Intel/neural-chat-7b-v3-3'

#### Further instrcutions here..........
conv_mode = 'llava_v1'
disable_torch_init()

model_name = get_model_name_from_path(model_path)
print('model name', model_name)
print('model base', model_base)


tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Replace 'YOUR_API_KEY' with your actual OpenAI API key

# Initialize the conversation history as an empty list
conversation_history = []
chatbot_response=[]
image_mem = None
# Function to use GPT-3 as the chatbot
def clear_history(chat_history):
    global conversation_history, chatbot_response, image_mem
    conversation_history = []
    chatbot_response = []
    image_mem = None
    return [], None
def add_text(text,chat_history):
    chat_history.append((text, None))
    return '', chat_history
def add_file(chat_history, file):
    global image_mem
    chat_history.append(((file.name,), None))
    #process the images here
    image_mem = Image.open(file.name)
    return chat_history
def chat_with_gpt(text, chat_history):
    global conversation_history,chatbot_response, image_mem # Use the global conversation_history variable
    # Prepare the input text, adding image-related tokens if needed
    cur_prompt = chat_history[-1][0]
    if not image_mem:
        chat_history.append((None, "Upload an image first so you can ask about it 😊"))
        return "", chat_history
    if image_mem:
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
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    response = outputs.strip()

    #print(response)

    chat_history.append((None, response))
    return "", chat_history
def example_add(msg, file,chat_history):
    global image_mem
    image_mem = Image.open(file)
    chat_history.append(((file,), None))
   # process the images here
   #  add_text(msg, chat_history)
   #  a, chat_history = chat_with_gpt(msg, chat_history)
    return chat_history



title = "RS-LLaVA (v1)-Demo"
description = "RS-LLaVA (v1) is pretrained and finetuned using LLaVA data and enriched with RS instruction data. It uses Intel/neural-chat-7b-v3-3 as LLM and CLIP-14 as vission Encoder." \
              "\n BigData@AI, Computer Enginnering, CCIS, King Saud University."

# Adjusting the interface layout
with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False) as demo:
    with gr.Column():

        # Image upload column on the left, smaller size
        with gr.Row():
            with gr.Column():
                #gr.Markdown("<img src='https://kscdr.org.sa/sites/default/files/2021-08/KSU-1.png'>",scale=3)
                #gr.Markdown(f"# {title}\n{description}", scale=3)
                gr.Markdown(f"# {title}\n{description}")
                curr_img = gr.Image(label="Uploaded Image", interactive=False, type="filepath", scale=6)
                #btn = gr.UploadButton("Click to upload Image 🖼️", file_types=["image"], accepts_multiple_files=False)
                btn = gr.UploadButton("Click to upload Image 🖼️", file_types=["image"])
                clear = gr.Button("Clear Chat 🗑️")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(bubble_full_width=True, label="Dialogue", height=800)
                msg = gr.Textbox(label="Enter your question here", placeholder="Type your question and hit enter")
                sub = gr.Button("Submit")
        # Chat window column on the right, larger size
        ex = gr.Examples([['Are there any spectators present in the stands?', 'ex_img/Baseball.png'],
                              ['what is the color of the two cars?', 'ex_img/CarRacing.png'],
                              ['what is the weather condition?', 'ex_img/Cycling .png']]
                             , [msg, curr_img, chatbot], chatbot, example_add, run_on_click=True)

    # Bind functions to widgets
    msg.submit(add_text, [msg, chatbot], [msg, chatbot]).then(fn=chat_with_gpt,inputs=[msg,chatbot], outputs=[msg,chatbot])
    clear.click(clear_history, [], [chatbot, curr_img])
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(lambda a: image_mem.resize((int(image_mem.size[0]*300/image_mem.size[1]), 300)), outputs=[curr_img])
    sub.click(add_text, [msg, chatbot], [msg, chatbot]).then(fn=chat_with_gpt,inputs=[msg,chatbot], outputs=[msg,chatbot])
    # Optional: Add a footer for credits or additional information
    gr.Markdown("---\n© 2024 BigData@AI Team: Computer Engineering Department-CCIS, King Saud University")

if __name__ == "__main__":
    demo.launch()

