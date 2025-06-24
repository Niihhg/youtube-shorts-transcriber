from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import re
import datetime
from pathlib import Path

# ===== 設定 =====
TRANSCRIPT_DIR = Path("C:/Users/Debor/Desktop/Trésor Lexical/00_YouTube_Shorts_Transcripts")
OUTPUT_DIR = Path("C:/Users/Debor/Desktop/Trésor Lexical/10_VIDEOS")
PROMPT_PATH = "prompts/video_note_prompt.txt"  # あなたのGPTプロンプトを記載した外部ファイル
MODEL = "gpt-4.1"
SYSTEM_PROMPT = "あなたはVideo Note Creator for Obsidianです。"

# ===== 修正済スクリプトの最新ファイルを取得 =====
if not TRANSCRIPT_DIR.exists():
    print(f"❌ エラー: ディレクトリ {TRANSCRIPT_DIR} が存在しません。")
    print("ディレクトリパスを確認してください。")
    exit(1)

md_files = list(TRANSCRIPT_DIR.glob("*.md"))
if not md_files:
    print(f"❌ エラー: {TRANSCRIPT_DIR} にMarkdownファイルが見つかりません。")
    print("先にYouTube動画の文字起こしファイルを作成してください。")
    exit(1)

latest_file = max(md_files, key=os.path.getmtime)
print(f"📄 処理対象ファイル: {latest_file.name}")

with open(latest_file, "r", encoding="utf-8") as f:
    transcript = f.read()

# ===== プロンプトを組み立てる =====
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    base_prompt = f.read()

user_prompt = base_prompt.replace("{transcript}", transcript)

# ===== GPT呼び出し =====
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.4
)

note_output = response.choices[0].message.content

# ===== ファイル名slugを抽出する関数 =====
def extract_slug_from_video_note(note_text):
    match = re.search(r"ファイル名slug:\s*([a-z0-9_]{4,60})", note_text)
    if match:
        return match.group(1)
    # なければデフォルト
    return "video_note"

# ===== ファイル名slugを取得 =====
filename_title = extract_slug_from_video_note(note_output)

# ===== slug行をノート本文から削除 =====
note_output = re.sub(r"\n?ファイル名slug:[^\n]*", "", note_output)
today = datetime.date.today().isoformat()
output_filename = f"{filename_title}.md"
output_path = OUTPUT_DIR / output_filename

# ===== 出力ディレクトリを作成 =====
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 既存YAMLを検出してdateだけ追加 =====
if note_output.strip().startswith("---"):
    parts = note_output.strip().split("---")
    if len(parts) >= 3:
        header = parts[1].strip()
        body = "---".join(parts[2:]).strip()
        if "date:" not in header:
            header += f"\ndate: {today}"
        final_output = f"---\n{header}\n---\n\n{body}"
    else:
        final_output = note_output
else:
    final_output = note_output

with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_output)

print(f"✅ Videoノートを {output_path} に保存しました（元: {latest_file.name}）")