import asyncio
import json
import sys
#from transformers import T5ForConditionalGeneration, AutoTokenizer
#g2p_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
#g2p_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
#from g2p import make_g2p
#transducer = make_g2p('eng', 'eng-arpabet')
from g2p_en import G2p
g2p_model = G2p()
import re
import html

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

# For local streaming, the websockets are hosted without ssl - ws://
HOST = 'localhost:5005'
URI = f'ws://{HOST}/api/v1/chat-stream'
CLIENT_URI = 'ws://localhost:5006/input'
TTS_URI = 'ws://localhost:5007/input'

from phonemizer import phonemize

def get_phones(text):
#    return phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    return g2p_model(text)
#phones += [  transducer(text).output_string ]

def g2p(text):
    words = []
    # tokenized English words
    punc = list(re.finditer(r"([ \.,?!:\"。？！，“‘])", text))
    phones = []
    if len(punc) == 0:
        phones += get_phones(text)     
    else:
        i = 0
        for m in punc:
            seg = text[i:m.start()]
            seg_phones = get_phones(seg)
            print(f"got {seg_phones} for {seg}")
            phones += seg_phones
            
            if m.group(1) != " ":
                phones += [ m.group(1) ]
            i = m.end()
        if punc[-1].start() != len(text) - 1:
            seg = text[punc[-1].end():]
            seg_phones = get_phones(seg)
            print(f"last seg_phones {seg_phones}")
            phones += seg_phones
    print(f"Phonemized {text} to {phones}")
    return " ".join(phones)

#print(g2p("I am, but it doesn't seem like enough?"))
#sys.exit()
async def run(user_input, history):
    print(f"RUnning with history {history}")
    # Note: the selected defaults change from time to time.
    request = {
        'user_input': user_input,
        'max_new_tokens': 250,
        'auto_max_new_tokens': False,
        'history': history,
        'mode': 'chat',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'character': 'Example',
        'instruction_template': 'Vicuna-v1.1',  # Will get autodetected if unset
        'your_name': 'You',
        # 'name1': 'name of user', # Optional
        # 'name2': 'name of character', # Optional
        # 'context': 'character context', # Optional
        # 'greeting': 'greeting', # Optional
        # 'name1_instruct': 'You', # Optional
        # 'name2_instruct': 'Assistant', # Optional
        # 'context_instruct': 'context_instruct', # Optional
        # 'turn_template': 'turn_template', # Optional
        'regenerate': False,
        '_continue': False,
        'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    async with websockets.connect(URI, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))

        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data["history"]
                case 'stream_end':
                    return

async def handler(websocket):
    history = {'internal': [], 'visible': []}
    cur_sentence = []

    async with websockets.connect(TTS_URI, ping_interval=None) as ttssocket:
        while True:
            user_input = await websocket.recv()
            user_input = json.loads(user_input)
            cur_len = 0
            async for new_history in run(user_input["input"], history):
                cur_message = new_history['internal'][-1][1][cur_len:].replace(":"," ").replace("-", "")
                is_eos = re.search("[\.?!。？！\n]$", cur_message) is not None
                cur_len += len(cur_message) 
                cur_sentence += [ cur_message ]
                if is_eos:
                    verbalize = re.sub("\*.+\*", "", "".join(cur_sentence))
                    phones = g2p(verbalize)
                    await ttssocket.send("".join(phones))
                    tts_response = await ttssocket.recv()
                    tts_response=json.loads(tts_response)
                    packet = json.dumps({"message":"".join(cur_sentence), "phones":"".join(phones), "audio":tts_response["audio"], "durations":tts_response["durations"]})
                    await websocket.send(packet)
                    cur_sentence = []
                    cur_phones = []
            history = new_history
async def main():
    async with websockets.serve(handler, "", 5006):
        await asyncio.Future()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
