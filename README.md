# GPU対応チャットボットCLI

NVIDIA GPUとsarashina2.2-0.5B日本語モデルを使用した超軽量チャットボット

## 特徴

- **超軽量モデル**: sarashina2.2-0.5B (0.5Bパラメータ、日本語特化)
- **GPU加速**: NVIDIA CUDA対応
- **オープンソース**: Apache 2.0ライセンス
- **シンプルなCLI**: 使いやすいコマンドラインインターフェース

## 必要要件

- Python 3.8以上
- NVIDIA GPU (CUDA対応)
- 1GB以上のVRAM推奨

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

**sarashina2.2-0.5B-instruct-v0.1**
- パラメータ数: 0.5B (超軽量)
- 日本語特化モデル
- ライセンス: Apache 2.0
- 開発元: SB Intuitions

## カスタマイズ

[chatbot.py](chatbot.py)を編集して以下を変更可能:

- `max_length`: 応答の最大長
- `temperature`: 生成の多様性 (0.0-1.0)
- `model_name`: 他の軽量モデルに変更可能

### 他の推奨軽量モデル

- `sbintuitions/sarashina2.2-1B-instruct-v0.1` (1B、より高性能な日本語モデル)
- `microsoft/Phi-3.5-mini-instruct` (3.8B、多言語対応)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B、英語)
