import os, sys, yaml, datetime, pathlib
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

CONFIG_FILE = 'config.yaml'
if not pathlib.Path(CONFIG_FILE).exists():
    print(f"❌ {CONFIG_FILE} がありません。")
    sys.exit(1)

cfg = yaml.safe_load(open(CONFIG_FILE, 'r', encoding='utf-8'))
vault_path = pathlib.Path(cfg['vault_path']).expanduser()
vault_path.mkdir(parents=True, exist_ok=True)
model_name = cfg.get('whisper_model', 'base')

url = input('Paste YouTube Shorts URL: ').strip()
if not url:
    print('❌ URL が空です'); sys.exit(1)

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': '%(id)s.%(ext)s',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
    'quiet': True,
}
with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    wav = pathlib.Path(f"{info['id']}.wav")

print('文字起こし中…')
model = WhisperModel(model_name)
segments, _ = model.transcribe(wav)

md_lines = [url, ""]

for segment in segments:
    speaker = segment.speaker if hasattr(segment, "speaker") else "Speaker"
    text = segment.text.strip()
    md_lines.append(f"[{speaker}] {text}")

out_file = vault_path / f"{info['id']}.md"
out_file.write_text("\n".join(md_lines), encoding="utf-8")
print(f'✅ Done! Saved Markdown to {out_file}')
