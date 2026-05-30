import sys
import mlx_whisper

if len(sys.argv) < 2:
    print("Usage: python test_mlx.py <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]
print(f"Transcribing {audio_file}...")
result = mlx_whisper.transcribe(
    audio_file, 
    path_or_hf_repo="mlx-community/whisper-tiny", 
    word_timestamps=True
)

print(result.keys())
print("First segment:")
print(result["segments"][0])
