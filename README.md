# YouTube Shorts Transcriber + Video Note Generator

YouTube Shortsから高精度な文字起こしを行い、さらにGPT-4.1を使用してObsidian用の詳細なビデオノートを生成するツールセットです。

---

## 🔧 機能概要

### 1. 音声文字起こし機能（run.py）
- YouTube URLから音声を自動ダウンロード（yt-dlp使用）
- WhisperX による高精度音声認識（Largeモデル使用）
- Resemblyzer による話者分離・識別
- Obsidian Vault内の指定フォルダに`.md`形式で出力

### 2. ビデオノート生成機能（video_note_generator.py）
- 文字起こしファイルを読み込み
- GPT-4.1を使用してフランス語学習用の詳細なノートを生成（GPT-4ではパワー不足）
- 語彙表、文法解説、文化的背景などを含む包括的な学習資料を作成

---

## 📁 ファイル構成

```
youtube-shorts-transcriber/
├── run.py                      ← メイン文字起こしスクリプト
├── video_note_generator.py     ← GPTビデオノート生成スクリプト
├── config.yaml                 ← 設定ファイル
├── requirements.txt            ← 必要ライブラリ一覧
├── prompts/
│   └── video_note_prompt.txt   ← GPT用プロンプトテンプレート
├── .env                        ← OpenAI APIキー設定
├── README.md                   ← この説明ファイル
├── .gitignore                  ← 除外設定
└── .venv/                      ← 仮想環境（Git管理対象外）
```

---

## 🚀 使い方

### 1. 初期セットアップ

#### 仮想環境の作成とライブラリインストール
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows の場合
pip install -r requirements.txt
```

#### 設定ファイルの準備

**config.yaml を作成：**
```yaml
vault_path: "C:\\Users\\あなたの名前\\Desktop\\Trésor Lexical"
subdir: ""
whisper_model: large-v3
num_speakers_range: "2,4"
```

**.env ファイルを作成：**
```
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token
```

### 2. 文字起こしの実行

```bash
python run.py
```

実行後、YouTube URLの入力を求められます。処理完了後、以下の場所にファイルが出力されます：
```
[VaultPath]/00_YouTube_Shorts_Transcripts/[動画タイトル].md
```

### 3. ビデオノートの生成

```bash
python video_note_generator.py
```

最新の文字起こしファイルを自動で読み込み、GPT-4.1を使用して詳細な学習ノートを生成します。出力先：
```
[VaultPath]/10_VIDEOS/[生成されたslug].md
```

---

## 📝 出力内容例

### 文字起こしファイル（run.py）
```markdown
---
title: 動画タイトル
source: "https://youtube.com/shorts/xxxxx"
created: 2025-01-15 12:00:00
tags: [transcript, youtube-shorts]
---

## Transcript

> **SPEAKER_1**: こんにちは、今日のテーマは…
> **SPEAKER_2**: はい、それについて説明します。
```

### ビデオノートファイル（video_note_generator.py）
```markdown
---
tags:
  - "#vidéo/conversation"
  - "#vidéo/culture"
source: "https://youtube.com/shorts/xxxxx"
---

# 🎬 タイトル

## 📝 要約
動画内容の日本語要約...

## 🎙️【文字起こし（全文）】
> *Bonjour, comment allez-vous ?*
> **こんにちは、お元気ですか？**

## ✨ ちょっと面白い豆知識
文化的背景や語源の解説...

## 🇫🇷 フランス語ミニノート：YouTubeショートから学ぶ B1＋表現集
語彙表と表現フレーズの詳細解説...

## 📘 文法ポイント
具体的な文法解説...

## 🪄 追加の視点・論点・注意点
学習者向けの洞察...
```

---

## 📦 動作要件

- Python 3.9 以上
- ffmpeg（音声変換に必要）
- CUDA対応GPU（推奨、CPUでも動作可能）
- インターネット接続
- OpenAI API キー
- Hugging Face トークン（話者分離用）
- Obsidian（出力ファイルを閲覧・活用するため）

---

## ⚙️ 設定項目

### config.yaml
- `vault_path`: Obsidian Vaultのパス
- `whisper_model`: 使用するWhisperモデル（large-v3推奨）
- `hf_token`: Hugging Face認証トークン
- `num_speakers_range`: 話者数の範囲

### .env
- `OPENAI_API_KEY`: OpenAI APIキー（GPT-4.1アクセス用）

---

## ⚠️ 注意事項

- `.venv/`, `.env`, `config.yaml`は個人環境依存のためGitには含めません
- 音声ファイル（`.wav`, `.mp4`）や実行ファイル（`.exe`）は`.gitignore`で除外されています
- OpenAI APIの使用には料金が発生する場合があります
- Hugging Face トークンは話者分離機能に必要です

---

## 🔄 ワークフロー

1. **run.py** でYouTube動画から文字起こしを生成
2. **video_note_generator.py** で文字起こしを基に詳細な学習ノートを作成
3. Obsidianで生成されたノートを活用して学習

---

## 👤 作者
Nicco  
(日本語／英語／フランス語対応)









