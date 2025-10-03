# GPU対応チャットボットCLI

NVIDIA GPUとPhi-3-mini (3.8B)モデルを使用した軽量チャットボット

## 特徴

- **軽量モデル**: Microsoft Phi-3-mini (3.8Bパラメータ)
- **GPU加速**: NVIDIA CUDA対応
- **オープンソース**: MITライセンスのモデル使用
- **シンプルなCLI**: 使いやすいコマンドラインインターフェース

## 必要要件

- Python 3.8以上
- NVIDIA GPU (CUDA対応)
- 8GB以上のVRAM推奨

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python chatbot.py
```

### コマンド

- メッセージ入力: 質問や指示を入力してEnter
- 終了: `exit`, `quit`, または `q` を入力

## モデルについて

**Phi-3-mini-4k-instruct**
- パラメータ数: 3.8B
- コンテキスト長: 4K トークン
- ライセンス: MIT
- 開発元: Microsoft

## カスタマイズ

[chatbot.py](chatbot.py)を編集して以下を変更可能:

- `max_length`: 応答の最大長
- `temperature`: 生成の多様性 (0.0-1.0)
- `model_name`: 他の軽量モデルに変更可能

### 他の推奨軽量モデル

- `microsoft/Phi-3-mini-128k-instruct` (128Kコンテキスト版)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B、より軽量)
- `stabilityai/stablelm-2-zephyr-1_6b` (1.6B)
