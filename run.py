#!/usr/bin/env python3
"""
run_resemble.py — YouTube Shorts → 高精度音声認識＋Resemblyzer話者識別版
"""

from __future__ import annotations
import sys, os, re, pathlib, tempfile, datetime, warnings, yaml
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import whisperx
import yt_dlp
from pydub import AudioSegment
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")

# --- 設定 ---
MODEL = "large-v3"
LANGUAGE = None

# 設定ファイルの読み込み
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("❌ config.yamlファイルが見つかりません")
    sys.exit(1)
except yaml.YAMLError:
    print("❌ config.yamlファイルの形式が正しくありません")
    sys.exit(1)

# ObsidianのVaultパスを設定
VAULT_PATH = pathlib.Path(config["vault_path"])
# 出力先フォルダの設定
OUTPUT_DIR = VAULT_PATH / "00_YouTube_Shorts_Transcripts"

# 出力ディレクトリが存在しない場合は作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- YouTube 音声取得 ---
def download_audio(url: str, out_dir: pathlib.Path) -> Tuple[pathlib.Path, Dict[str, Any]]:
    try:
        opts = {
            'format': 'bestaudio/best',
            'outtmpl': f"{out_dir}/%(id)s.%(ext)s",
            'postprocessors': [{
                'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'
            }],
            'quiet': True
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        wav_path = out_dir / f"{info['id']}.wav"
        if not wav_path.exists():
            raise FileNotFoundError("音声ファイルのダウンロードに失敗しました")
        return wav_path, info
    except Exception as e:
        print(f"❌ 音声のダウンロード中にエラーが発生しました: {e}")
        sys.exit(1)

# --- WhisperX 音声認識 ---
def transcribe_whisperx(wav: pathlib.Path) -> Tuple[Dict[str, Any], torch.device]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisperx.load_model(MODEL, device, compute_type="float16" if device=="cuda" else "int8")
        audio = whisperx.load_audio(str(wav))
        result = model.transcribe(audio, language=LANGUAGE)
        align_model, metadata = whisperx.load_align_model(result["language"], device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, device=device)
        return aligned, device
    except Exception as e:
        print(f"❌ 音声認識中にエラーが発生しました: {e}")
        sys.exit(1)

# --- 話者識別（Resemblyzer） ---
def diarize_resemblyzer(wav_file: pathlib.Path, segments: list[dict]) -> list[str]:
    try:
        wav = preprocess_wav(str(wav_file))
        encoder = VoiceEncoder()
        embeds = []
        mids = []
        for seg in segments:
            start, end = seg["start"], seg["end"]
            segment = wav[int(start * 16000):int(end * 16000)]
            if len(segment) < 1600: continue
            embed = encoder.embed_utterance(segment)
            embeds.append(embed)
            mids.append((start + end) / 2)

        if len(embeds) < 2:
            return ["SPEAKER_1"] * len(segments)

        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.65).fit(embeds)
        labels = clustering.labels_
        spk_map = {i: f"SPEAKER_{i+1}" for i in set(labels)}
        assigned = []
        idx = 0
        for seg in segments:
            if idx < len(labels):
                seg["speaker"] = spk_map[labels[idx]]
                idx += 1
            else:
                seg["speaker"] = "SPEAKER_UNKNOWN"
        return [seg["speaker"] for seg in segments]
    except Exception as e:
        print(f"❌ 話者識別中にエラーが発生しました: {e}")
        return ["SPEAKER_1"] * len(segments)

# --- Markdown 出力 ---
def generate_markdown(result: Dict[str, Any], url: str, title: str) -> str:
    lines = [
        "---",
        f"title: {title}",
        f"source: \"{url}\"",
        f"created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "tags: [transcript, youtube-shorts]",
        "---",
        "",
        "## Transcript",
        ""
    ]
    prev = None
    buf = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "").strip()
        if not text: continue
        if speaker != prev:
            if prev and buf:
                lines.append(f"> **{prev}**: {' '.join(buf)}")
            prev = speaker
            buf = [text]
        else:
            buf.append(text)
    if prev and buf:
        lines.append(f"> **{prev}**: {' '.join(buf)}")
    return "\n".join(lines)

# --- メイン ---
def main():
    try:
        url = input("YouTube Shorts URL: ").strip()
        if not url:
            print("❌ URLが入力されていません")
            sys.exit(1)
            
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            wav_path, info = download_audio(url, tmp_path)
            result, _ = transcribe_whisperx(wav_path)
            diarize_resemblyzer(wav_path, result["segments"])
            
            # タイトルの取得と整形
            title = info.get("title", "video")
            safe_title = re.sub(r"[<>:/\\|?*\"]", "", title)
            
            # Markdownの生成
            md = generate_markdown(result, url, safe_title)
            
            # ファイル名の作成（タイトルを含む）
            filename = f"{safe_title}.md"
            output_path = OUTPUT_DIR / filename
            
            # Markdownファイルの保存
            output_path.write_text(md, encoding="utf-8")
            print(f"✅ 出力完了: {output_path}")
            
    except KeyboardInterrupt:
        print("\n⚠️ 処理を中断しました")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
