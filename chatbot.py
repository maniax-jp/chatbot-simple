#!/usr/bin/env python3
"""
NVIDIA GPU対応チャットボットCLI
日本語特化の超軽量sarashina2.2-0.5Bモデルを使用
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys


class ChatBot:
    def __init__(self, model_name="sbintuitions/sarashina2.2-0.5B-instruct-v0.1"):
        """
        チャットボットの初期化

        Args:
            model_name: 使用するHugging Faceモデル名
        """
        print(f"モデルをロード中: {model_name}")
        print("初回実行時はモデルのダウンロードに時間がかかります...")

        # GPUが利用可能か確認
        if not torch.cuda.is_available():
            print("警告: NVIDIA GPUが検出されませんでした。CPUで実行します。")
            self.device = "cpu"
        else:
            self.device = "cuda"
            print(f"GPU検出: {torch.cuda.get_device_name(0)}")

        # トークナイザーとモデルのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation="eager",
            use_cache=False  # キャッシュを無効化してseen_tokensエラーを回避
        )

        # テキスト生成パイプラインの作成
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print("モデルのロードが完了しました！\n")

    def chat(self, user_input, max_length=512, temperature=0.7):
        """
        ユーザー入力に対して応答を生成

        Args:
            user_input: ユーザーからの入力テキスト
            max_length: 生成する最大トークン数
            temperature: 生成の多様性（0.0-1.0）

        Returns:
            生成された応答テキスト
        """
        # Phi-3用のプロンプトフォーマット
        messages = [
            {"role": "user", "content": user_input}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 応答生成
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            return_full_text=False,
            use_cache=False  # キャッシュを無効化
        )

        response = outputs[0]["generated_text"]
        return response.strip()

    def run_interactive(self):
        """対話型CLIセッションを実行"""
        # モデル名を取得（パス形式から名前部分のみ抽出）
        model_display_name = self.model.config._name_or_path.split('/')[-1]

        print("=" * 60)
        print(f"チャットボットCLI - {model_display_name}")
        print("=" * 60)
        print("終了するには 'exit', 'quit', 'q' を入力してください")
        print("=" * 60)
        print()

        while True:
            try:
                # ユーザー入力を取得
                user_input = input("あなた: ").strip()

                # 終了コマンドチェック
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nチャットを終了します。")
                    break

                # 空入力をスキップ
                if not user_input:
                    continue

                # 応答生成
                print("ボット: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nチャットを終了します。")
                break
            except Exception as e:
                print(f"\nエラーが発生しました: {e}")
                print("続行します...\n")


def main():
    """メイン関数"""
    try:
        # チャットボット初期化
        bot = ChatBot()

        # 対話型セッション開始
        bot.run_interactive()

    except KeyboardInterrupt:
        print("\n\n終了します。")
        sys.exit(0)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
