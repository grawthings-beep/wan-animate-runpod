# RunPodテンプレート設定

## 1. GitHub側

1. `main` のGitHub Actions `Build GHCR image` が成功していることを確認する。
2. GitHubの **Packages → wan-animate-runpod → Package settings** を開く。
3. RunPodから認証なしで使うならvisibilityを **Public** にする。

## 2. tokenをRunPod Secretに登録

1. CivitAIでAPI tokenを発行する。
2. RunPod Consoleの **Secrets** で `CIVITAI_TOKEN` を作る。
3. Hugging Face read tokenをSecret `HF_TOKEN` として作る。
4. tokenを平文の環境変数へ直接貼らない。

## 3. Pod Template

```text
Template type: Pod
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
Container disk: 50 GB
Volume disk: 250 GB以上
Volume mount path: /workspace
Expose HTTP port: 8188
```

環境変数:

```text
PORT=8188
LISTEN=0.0.0.0
DOWNLOAD_MODELS=1
MODEL_PROFILE=full
CIVITAI_API_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
RUN_DEP_CHECK=1
DOWNLOAD_WORKERS=4
ARIA2_CONNECTIONS=8
ARIA2_SPLITS=8
HF_XET_HIGH_PERFORMANCE=1
COMFYUI_ARGS=--reserve-vram 3
```

Web画面では `CIVITAI_API_TOKEN` の値で鍵アイコンから `CIVITAI_TOKEN`、`HF_TOKEN` の値で鍵アイコンから `HF_TOKEN` を選びます。`MODEL_MANIFEST_URL` は追加しません。

## 4. GPU

- 最高品質: A100 80GB / H100 80GB
- 実用ライン: L40S / RTX 6000 Ada 48GB
- 24GB: CPUオフロードが増え、高解像度動画では非推奨

## 5. 初回起動

1. Podを起動する。
2. Logsで `MODEL PROFILE: full`（ループ専用なら `loop-quality`）、`DOWNLOAD:`、最後の `missing=0` を確認する。
3. 初回のみ約98GBを取得する。`TRANSFER ENGINE: 4 files in parallel` を確認する。Podを止めてもHFキャッシュまたは `.part` から再開する。
4. **Connect → HTTP Service [Port 8188]** でComfyUIを開く。
5. **Workflows → Open** からAIOまたはSeamless Loopを選ぶ。

## 6. 更新

新しいGitHub Actionsビルド後にPodを再作成しても、`/workspace` のモデル、入力、出力、ユーザーワークフローは残ります。同名のユーザーワークフローは上書きしません。標準のモデルmanifestだけは新イメージの検証済み版へ自動更新されます。
