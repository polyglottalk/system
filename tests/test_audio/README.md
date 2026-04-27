# tests/test_audio/README.md
# Place a 2.5-second 16 kHz mono WAV file here named `hello.wav`.
# The file should contain the spoken phrase "Hello, how are you?"
# in English.  It is used by tests/test_asr.py.
#
# To create one with ffmpeg (if you have it installed):
#   ffmpeg -f lavfi -i "sine=frequency=440:duration=2.5" -ar 16000 -ac 1 hello.wav
# (This produces a tone, not real speech — use a real recording for accuracy testing.)
