import sounddevice as sd
from tortoise.api import MODELS_DIR
# from tortoise.api import TextToSpeech
from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voices
import time

tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)
voice_samples, conditioning_latents = load_voices(['geralt'])
#
# audio = tts.tts("What are you up to today?")
# sd.play(audio[0][0], samplerate=24000)
#
# time.sleep(4)
#
#

# audio, dbg_state = tts.tts_with_preset("This is what I want you to say", k=1, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset="fast", use_deterministic_seed=1, return_deterministic_state=True, cvvp_amount=.0)
# sd.play(audio[0][0], samplerate=24000)
#
# time.sleep(4)


# stream = tts.tts_stream("This is my hello world example", voice_samples=voice_samples, conditioning_latents=conditioning_latents, verbose=True, stream_chunk_size=40)
#
# for audio_chunk in stream:
#     print(audio_chunk)
#     sd.play(audio_chunk.cpu(), samplerate=24000)
#     sd.wait()
#     # for chunk in audio_chunk:
#     #     audio = chunk.cpu().numpy().flatten()
#     #     sd.play(audio, samplerate=24000)


def run(text, chunk_size):
    stream = tts.tts_stream(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, verbose=True, stream_chunk_size=chunk_size)
    for audio_chunk in stream:
        sd.play(audio_chunk.cpu(), samplerate=24000)
        sd.wait()

def main():
    run("This is the first test, just wait until we get to the second one", 256)
