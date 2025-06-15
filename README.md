# YouTube Shorts Transcriber

YouTube Shorts の動画から音声を抽出し、WhisperX + Resemblyzer によって高精度な文字起こしと話者識別を行い、Obsidian用のMarkdownファイルとして出力します。

---

## 🔧 機能概要

- YouTube URL から音声を自動ダウンロード（yt-dlp 使用）
- WhisperX による音声認識と字幕生成
- Resemblyzer による話者分離
- Obsidian Vault 内の指定フォルダに `.md` 形式で出力

---

## 📁 ファイル構成

youtube-shorts-transcriber/
├── run.py ← メインスクリプト
├── config.yaml ← VaultPath を指定
├── requirements.txt ← 必要なライブラリ一覧
├── README.md ← この説明ファイル
├── .gitignore ← 除外設定
└── .venv/ ← 仮想環境（Git管理対象外）

yaml
Copier
Modifier

---

## 🚀 使い方

1. 仮想環境を作成し、必要ライブラリをインストール

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows の場合
pip install -r requirements.txt
config.yaml を作成し、VaultPath を指定

yaml
Copier
Modifier
VaultPath: C:/Users/あなたの名前/Obsidian/MyVault
スクリプトを実行し、YouTube URL を入力

bash
Copier
Modifier
python run.py
実行後、Vault 内の 00_YouTube_Shorts_Transcripts フォルダに .md が出力されます。

📝 出力先と内容例
出力ファイルは以下のように保存されます：

swift
Copier
Modifier
C:/Users/あなたのVault名/00_YouTube_Shorts_Transcripts/
└── [動画タイトル].md
Markdownの内容例：

markdown
Copier
Modifier
---
title: [動画タイトル]
source: "https://youtube.com/shorts/xxxxx"
created: 2025-06-15 12:00:00
tags: [transcript, youtube-shorts]
---

## Transcript

> **SPEAKER_1**: こんにちは、今日のテーマは…
> **SPEAKER_2**: はい、それについて説明します。
📦 動作要件
Python 3.9 以上

ffmpeg（音声変換に必要）

インターネット接続

Obsidian（出力ファイルを閲覧・活用するため）

⚠️ 注意事項
.venv/ や .wav, .mp4, .exe などの重いファイルは Git に含めないでください。

config.yaml は個人環境依存のため Git には含めず、.gitignore に追加してください。

👤 作者
Nico
(日本語／英語／フランス語対応)









