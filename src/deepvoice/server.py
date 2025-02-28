import sounddevice as sd
from tortoise.api import MODELS_DIR
from tortoise.api import TextToSpeech
from tortoise.api_fast import TextToSpeech as FastTextToSpeech
from tortoise.utils.audio import load_voices
import argparse
import socket



def handle_client(conn, addr, tts, voice_samples, conditioning_latents, chunk_size, realtime):
    print(f"Connected to {addr}")
    data = b""

    while True:
        chunk = conn.recv(1024)
        if not chunk:
            break
        data += chunk
        if b"EOF" in data:
            data = data.replace(b"EOF", b"")
            break

    received_text = data.decode()
    print(f"Received: {received_text}")
    chunked_text = received_text.split("\n")

    for text in chunked_text:

        if realtime:
            stream = tts.tts_stream(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, verbose=True, stream_chunk_size=chunk_size)
            for audio_chunk in stream:
                sd.play(audio_chunk.cpu(), samplerate=24000)
                sd.wait()
        else:
            audio, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset="fast", use_deterministic_seed=1, return_deterministic_state=True, cvvp_amount=.0)
            sd.play(audio[0][0], samplerate=24000)
            sd.wait()



    # conn.sendall(data)
    #
    # conn.close()
    # print(f"Connection from {addr} closed")
    #
def start_server(args):
    if args.realtime:
        tts = FastTextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)
    else:
        tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)
    voice_samples, conditioning_latents = load_voices([args.voice])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen()
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr, tts, voice_samples, conditioning_latents, args.chunk_size, args.realtime)

def main():
    parser = argparse.ArgumentParser(description="DeepVoice Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9986)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--voice", type=str, default="geralt")
    parser.add_argument("--realtime", action="store_true", help="Run with FastTextToSpeech")
    args = parser.parse_args()

    start_server(args)

if __name__ == "__main__":
    main()
