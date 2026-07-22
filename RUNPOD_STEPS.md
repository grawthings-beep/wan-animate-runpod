# RunPodテンプレート設定

## 1. GitHub側

1. `main` のGitHub Actions `Build GHCR image` が成功していることを確認する。
2. GitHubの **Packages → wan-animate-runpod → Package settings** を開く。
3. RunPodから認証なしで使うならvisibilityを **Public** にする。

## 2. CivitAI tokenをRunPod Secretに登録

1. CivitAIでAPI tokenを発行する。
2. RunPod Consoleの **Secrets** で `civitai_api_token` を作る。
3. tokenを平文の環境変数へ直接貼らない。

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
CIVITAI_API_TOKEN={{ RUNPOD_SECRET_civitai_api_token }}
RUN_DEP_CHECK=1
ARIA2_CONNECTIONS=16
ARIA2_SPLITS=16
COMFYUI_ARGS=--reserve-vram 3
```

`MODEL_MANIFEST_URL` と `HF_TOKEN` は空欄で構いません。

## 4. GPU

- 最高品質: A100 80GB / H100 80GB
- 実用ライン: L40S / RTX 6000 Ada 48GB
- 24GB: CPUオフロードが増え、高解像度動画では非推奨

## 5. 初回起動

1. Podを起動する。
2. Logsで `MODEL PROFILE: full`、`DOWNLOAD:`、最後の `missing=0` を確認する。
3. 初回のみ約98GBを取得する。Podを止めても `.part` から再開する。
4. **Connect → HTTP Service [Port 8188]** でComfyUIを開く。
5. **Workflows → Open** からAIOまたはSeamless Loopを選ぶ。

## 6. 更新

新しいGitHub Actionsビルド後にPodを再作成しても、`/workspace` のモデル、入力、出力、ユーザーワークフローは残ります。同名のユーザーワークフローは上書きしません。
