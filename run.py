#!/usr/bin/env python3
"""
run.py — YouTube Shorts → speaker-separated Markdown (Obsidian)

• WhisperX 3.3.4 で ASR → align
• pyannote/speaker-diarization-3.1 で 2-4 名を自動推定
• エラーハンドリングと後処理の改良版
"""

from __future__ import annotations
import sys, re, tempfile, pathlib, datetime, yaml, os
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

# ─── 1. 設定 ────────────────────────────────────
CFG_FILE = "config.yaml"
DEFAULT  = dict(
    vault_path    = "~/ObsidianVault",
    subdir        = "",
    whisper_model = "large-v3",
    language      = "auto",
    hf_token      = "",
    min_segment_duration = 0.20,  # 最小セグメント長（秒）
    merge_gap_threshold  = 0.30   # 同一話者の発話間隔しきい値（秒）
)

def load_config():
    """設定ファイルの読み込み・作成"""
    p_cfg = pathlib.Path(CFG_FILE)
    if not p_cfg.exists():
        p_cfg.write_text(yaml.dump(DEFAULT, allow_unicode=True, default_flow_style=False), encoding="utf-8")
        print("🔧 config.yaml を作成しました。HF_TOKENを設定して再実行してください。")
        print("   話者分離を使用する場合は、Hugging Face トークンが必要です。")
        sys.exit(0)
    
    try:
        cfg = {**DEFAULT, **yaml.safe_load(p_cfg.read_text(encoding="utf-8"))}
        return cfg
    except Exception as e:
        print(f"❌ config.yaml の読み込みに失敗: {e}")
        sys.exit(1)

cfg = load_config()
VAULT     = pathlib.Path(cfg["vault_path"]).expanduser()
VAULT.mkdir(parents=True, exist_ok=True)
TARGET    = VAULT / cfg["subdir"] if cfg["subdir"] else VAULT
TARGET.mkdir(parents=True, exist_ok=True)
MODEL     = cfg["whisper_model"]
LANG_OPT  = None if cfg["language"] in ("auto", "", None) else cfg["language"]
HF_TOKEN  = cfg.get("hf_token") or os.getenv("HF_TOKEN")
MIN_SPK, MAX_SPK = 2, 4
MIN_SEGMENT_DUR = cfg.get("min_segment_duration", 0.20)
MERGE_GAP = cfg.get("merge_gap_threshold", 0.30)

# ─── 2. YouTube → wav ───────────────────────────
def download_wav(url: str, out_dir: pathlib.Path) -> Tuple[pathlib.Path, Dict[str, Any]]:
    """YouTube動画から音声をWAV形式でダウンロード"""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("❌ yt-dlp が見つかりません: pip install yt-dlp")
        sys.exit(1)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{out_dir}/%(id)s.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noprogress': True,
        'no_warnings': True
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        
        if not info:
            raise Exception("動画情報の取得に失敗")
            
        wav_file = out_dir / f"{info['id']}.wav"
        if not wav_file.exists():
            raise Exception("WAVファイルの生成に失敗")
            
        return wav_file, info
        
    except Exception as e:
        print(f"❌ 音声ダウンロードエラー: {e}")
        print("   URLを確認するか、ffmpegがインストールされているか確認してください。")
        sys.exit(1)

# ─── 3. WhisperX ASR + Alignment ───────────────
def load_whisperx():
    """WhisperXモデルの読み込み"""
    try:
        import torch
        import whisperx
    except ImportError:
        print("❌ whisperx が見つかりません:")
        print("   pip install git+https://github.com/m-bain/whisperX.git@main")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        model = whisperx.load_model(MODEL, device, compute_type=compute_type)
        print(f"🎙  WhisperX model loaded: {MODEL} ({device})")
        return model, device
    except Exception as e:
        print(f"❌ WhisperX モデル読み込みエラー: {e}")
        print(f"   モデル '{MODEL}' が見つからない可能性があります。")
        sys.exit(1)

def transcribe_and_align(wav_file: pathlib.Path, language: Optional[str] = None) -> Dict[str, Any]:
    """音声認識とアライメント実行"""
    try:
        import whisperx
        import torch
    except ImportError:
        print("❌ whisperx のインポートに失敗")
        sys.exit(1)
    
    try:
        # 音声読み込み
        audio = whisperx.load_audio(str(wav_file))
        
        # ASRモデル読み込み・転写
        asr_model, device = load_whisperx()
        result = asr_model.transcribe(audio, language=language, batch_size=16)
        
        detected_lang = result.get("language", "en")
        print(f"🌍 検出言語: {detected_lang}")
        
        # アライメントモデル読み込み・実行
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang, 
                device=device
            )
            result = whisperx.align(result["segments"], align_model, metadata, audio, device=device)
            print("✓ 音声認識・アライメント完了")
        except Exception as e:
            print(f"⚠️  アライメント失敗: {e}")
            print("   基本的な転写結果のみ使用します")
        
        return result
        
    except Exception as e:
        print(f"❌ 音声認識エラー: {e}")
        sys.exit(1)

# ─── 4. Speaker Diarization ─────────────────────
def diarize_audio(wav_file: pathlib.Path):
    """話者分離の実行"""
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN が設定されていないため話者分離をスキップ")
        return None
    
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        print("❌ pyannote.audio が見つかりません: pip install pyannote.audio")
        return None
    
    try:
        # パイプライン読み込み
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        # GPU使用可能時はGPUに移動
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        
        print(f"👥 話者分離実行中... ({device})")
        
        # 話者分離実行（会話用に最適化）
        diarization = pipeline(
            str(wav_file),
            min_speakers=4,  # 4人の話者を強制
            max_speakers=4,  # 4人の話者に固定
            num_speakers=4,  # 話者数を明示的に指定
        )
        
        # 後処理: 短いセグメントの統合
        diarization = smooth_diarization(diarization)
        
        print("✓ 話者分離完了")
        return diarization
        
    except Exception as e:
        print(f"⚠️  話者分離失敗: {e}")
        print("   HF_TOKENが正しいか、ネットワーク接続を確認してください")
        return None

def smooth_diarization(diarization, min_duration: float = None, merge_gap: float = None):
    """話者分離結果の後処理（短いセグメント統合・ギャップ埋め）"""
    if min_duration is None:
        min_duration = 0.1  # より短いセグメントを許容
    if merge_gap is None:
        merge_gap = 0.2  # より短いギャップを許容
    
    # 短いセグメントを前後の長いセグメントに統合
    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append((segment, track, label))
    
    # 話者ごとのセグメントをグループ化
    speaker_segments = {}
    for segment, track, label in segments:
        if label not in speaker_segments:
            speaker_segments[label] = []
        speaker_segments[label].append((segment, track))
    
    # 各話者のセグメントを時間順にソート
    for label in speaker_segments:
        speaker_segments[label].sort(key=lambda x: x[0].start)
    
    # セグメントの統合
    filtered_segments = []
    for label, segs in speaker_segments.items():
        current_seg = None
        current_track = None
        
        for segment, track in segs:
            if current_seg is None:
                current_seg = segment
                current_track = track
            else:
                # ギャップが閾値以下の場合のみ統合
                if segment.start - current_seg.end <= merge_gap:
                    from pyannote.core import Segment
                    current_seg = Segment(
                        start=current_seg.start,
                        end=segment.end
                    )
                else:
                    filtered_segments.append((current_seg, current_track, label))
                    current_seg = segment
                    current_track = track
        
        if current_seg is not None:
            filtered_segments.append((current_seg, current_track, label))
    
    # 時間順にソート
    filtered_segments.sort(key=lambda x: x[0].start)
    return filtered_segments

# ─── 5. ASR + Diarization 統合 ─────────────────
def assign_speakers_to_words(asr_result: Dict[str, Any], diarization) -> Dict[str, Any]:
    """ASR結果に話者情報を割り当て"""
    if diarization is None:
        return asr_result
    
    try:
        import whisperx
        import numpy as np
        from pyannote.core import Segment
        
        # 話者セグメントを時間順にソート
        segments = []
        for segment, track, label in diarization:
            segments.append((segment, track, label))
        segments.sort(key=lambda x: x[0].start)
        
        # 話者ラベルの正規化（SPEAKER_1, SPEAKER_2, ...）
        speaker_map = {}
        for _, _, label in segments:
            if label not in speaker_map:
                speaker_map[label] = f"SPEAKER_{len(speaker_map) + 1}"
        
        # 各ASRセグメントに話者を割り当て
        for segment in asr_result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # 時間が重なる話者セグメントを探す
            overlapping_speakers = []
            for spk_seg, _, label in segments:
                if (spk_seg.start <= end_time and spk_seg.end >= start_time):
                    overlap = min(end_time, spk_seg.end) - max(start_time, spk_seg.start)
                    overlapping_speakers.append((label, overlap))
            
            if overlapping_speakers:
                # 最も重なりが大きい話者を選択
                speaker = max(overlapping_speakers, key=lambda x: x[1])[0]
                segment["speaker"] = speaker_map[speaker]
            else:
                # 最も近い話者を探す
                closest_speaker = None
                min_distance = float('inf')
                for spk_seg, _, label in segments:
                    distance = min(abs(spk_seg.end - start_time), abs(spk_seg.start - end_time))
                    if distance < min_distance:
                        min_distance = distance
                        closest_speaker = label
                
                if closest_speaker and min_distance < 0.5:  # 0.5秒以内の距離
                    segment["speaker"] = speaker_map[closest_speaker]
                else:
                    segment["speaker"] = "SPEAKER_UNKNOWN"
        
        print("✓ 話者-単語 割り当て完了")
        return asr_result
        
    except Exception as e:
        print(f"⚠️  話者割り当て失敗: {e}")
        print("   話者分離なしの結果を使用します")
        return asr_result

def process_audio(wav_file: pathlib.Path, language: Optional[str] = None) -> Dict[str, Any]:
    """音声処理のメインフロー"""
    # 1. 音声認識 + アライメント
    asr_result = transcribe_and_align(wav_file, language)
    
    # 2. 話者分離
    diarization = diarize_audio(wav_file)
    
    # 3. 結果統合
    final_result = assign_speakers_to_words(asr_result, diarization)
    
    return final_result

# ─── 6. Markdown 生成 ──────────────────────────
def generate_markdown(result: Dict[str, Any], original_url: str) -> str:
    """転写結果をMarkdown形式で生成"""
    lines = [
        f'source: "{original_url}"',
        "",
        "## Transcript",
        ""
    ]
    
    if not result.get("segments"):
        lines.extend(["> **Warning**: 転写結果が空です", ""])
        return "\n".join(lines)
    
    current_speaker = None
    current_text = []
    
    # セグメントを時間順にソート
    segments = sorted(result["segments"], key=lambda x: x["start"])
    
    for segment in segments:
        speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
        text = segment.get("text", "").strip()
        
        if not text:
            continue
        
        if speaker == current_speaker:
            # 同じ話者の場合は文章を連結
            current_text.append(text)
        else:
            # 話者が変わった場合、前の発話を出力
            if current_speaker is not None and current_text:
                lines.append(f"> **{current_speaker}**: {' '.join(current_text)}")
            
            # 新しい話者の発話を開始
            current_speaker = speaker
            current_text = [text]
    
    # 最後の発話を出力
    if current_speaker is not None and current_text:
        lines.append(f"> **{current_speaker}**: {' '.join(current_text)}")
    
    lines.append("")
    return "\n".join(lines)

# ─── 7. メイン処理 ─────────────────────────────
def validate_youtube_url(url: str) -> bool:
    """YouTube URLの妥当性をチェック"""
    youtube_patterns = [
        r"https?://(www\.)?youtube\.com/watch\?v=[\w-]+",
        r"https?://(www\.)?youtube\.com/shorts/[\w-]+",
        r"https?://youtu\.be/[\w-]+",
        r"https?://m\.youtube\.com/watch\?v=[\w-]+",
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)

def main():
    """メイン処理"""
    print("🎯 YouTube Shorts → Obsidian Markdown 変換ツール")
    print(f"📁 保存先: {TARGET}")
    print()
    
    # URL入力
    url = input("YouTube URL を入力してください: ").strip()
    
    if not url:
        print("❌ URLが入力されていません")
        sys.exit(1)
    
    if not validate_youtube_url(url):
        print("❌ 有効なYouTube URLを入力してください")
        print("   対応形式: youtube.com/watch, youtube.com/shorts, youtu.be")
        sys.exit(1)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # 1. 音声ダウンロード
            print("⬇️  音声ダウンロード中...")
            wav_file, video_info = download_wav(url, temp_path)
            
            video_duration = video_info.get('duration', 0)
            if video_duration > 300:  # 5分以上の場合警告
                print(f"⚠️  動画が長いです ({video_duration}秒)。処理に時間がかかる可能性があります。")
            
            # 2. 音声処理（ASR + 話者分離）
            print("🎙  音声認識・話者分離中...")
            result = process_audio(wav_file, LANG_OPT)
            
            # 3. Markdown生成
            markdown_content = generate_markdown(result, url)
            
            # 4. ファイル保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = video_info.get('id', 'unknown')
            output_filename = f"{timestamp}_{video_id}.md"
            output_path = TARGET / output_filename
            
            output_path.write_text(markdown_content, encoding="utf-8")
            
            print()
            print("✅ 処理完了!")
            print(f"📄 保存先: {output_path}")
            
            # 結果サマリー表示
            segment_count = len(result.get("segments", []))
            speakers = set(seg.get("speaker", "UNKNOWN") for seg in result.get("segments", []))
            print(f"📊 セグメント数: {segment_count}")
            print(f"👥 検出話者数: {len(speakers)} ({', '.join(sorted(speakers))})")
            
    except KeyboardInterrupt:
        print("\n⚠️  処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()