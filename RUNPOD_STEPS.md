# RunPodテンプレート設定

## 1. Secrets

RunPod ConsoleのSecretsで次を作ります。

```text
CIVITAI_TOKEN = CivitAI API tokenの実値
HF_TOKEN       = Hugging Face read tokenの実値
```

`loop-quality`だけならCivitAI tokenは不要です。他profileも使う場合だけ作成します。

Templateの環境変数では鍵アイコンを押し、次のように割り当てます。

```text
CIVITAI_API_TOKEN -> CIVITAI_TOKEN
HF_TOKEN           -> HF_TOKEN
```

tokenの実値を平文のEnvironment variablesへ貼らないでください。

## 2. Template

```text
Template type: Pod
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
Container disk: 50 GB
Volume disk: 100 GB以上（生成動画を多く残すなら150 GB推奨）
Volume mount path: /workspace
```

Networking configuration:

```text
HTTP Port label: ComfyUI
HTTP Port number: 8188
TCP Ports: 空欄
```

Environment variables:

```text
PORT=8188
LISTEN=0.0.0.0
DOWNLOAD_MODELS=1
MODEL_PROFILE=loop-quality
CIVITAI_API_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
RUN_DEP_CHECK=1
DOWNLOAD_WORKERS=4
ARIA2_CONNECTIONS=16
ARIA2_SPLITS=16
HF_SNAPSHOT_WORKERS=8
HF_XET_HIGH_PERFORMANCE=1
HF_XET_NUM_CONCURRENT_RANGE_GETS=64
HF_HUB_DOWNLOAD_TIMEOUT=300
COMFYUI_ARGS=--reserve-vram 3
```

RunPod UIではSecretを選ぶと表示形式が多少違う場合があります。重要なのは左側の環境変数名が`CIVITAI_API_TOKEN`/`HF_TOKEN`で、値が対応するSecretになっていることです。

## 3. GPU

- 第一候補: H100 80GB
- コスパ: A100 80GB
- 実用下限: L40SまたはRTX 6000 Ada 48GB

シームレスループはまず81 framesで継ぎ目を確認します。FIRST FRAMEとLAST FRAMEには同じ画像を設定し、RIFEは短いループが安定してから有効化します。

## 4. 初回起動

Logsで次を確認します。

```text
MODEL PROFILE: loop-quality (9 assets)
TRANSFER ENGINE: 4 files in parallel
...
[check_env] profile=loop-quality assets=9 missing=0
```

初回は約40.32 GBです。途中でPodを停止しても、同じVolumeならHF cacheまたは`.part`から再開します。`missing=0`になった後、ConnectからHTTP Service Port 8188を開きます。

## 5. workflowを開く

Workflowsから`wan22_smooth_v6_seamless_loop_runpod`を開きます。既存Volumeを新しいimageへ更新した場合は、末尾が`-bundle-<hash>`の最新版を開いてください。FIRST FRAMEとLAST FRAMEに同じ画像を設定し、最初は81 framesで生成してください。

## 6. 更新

新しいGitHub Actions buildが成功した後にPodを作り直します。`/workspace`のモデル、input、output、ユーザーworkflowはVolumeに残ります。bundle workflowが更新された場合も既存版を上書きせず、hash付きの新しい版が追加されます。
