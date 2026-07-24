# RunPodテンプレート設定

## 1. Secrets

RunPod ConsoleのSecretsで次を作ります。

```text
CIVITAI_TOKEN = CivitAI API tokenの実値
HF_TOKEN       = Hugging Face read tokenの実値
```

Templateの環境変数では鍵アイコンを押し、次のように割り当てます。

```text
CIVITAI_API_TOKEN -> CIVITAI_TOKEN
HF_TOKEN           -> HF_TOKEN
```

tokenの実値を平文のEnvironment variablesへ貼らないでください。

## 2. Template

```text
Template type: Pod
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:wan22-lightning-longvideo
Container disk: 50 GB
Volume disk: 200 GB以上（fullも保持するなら300 GB推奨）
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
MODEL_PROFILE=lightning-longvideo
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

Native Enhanced Lightningは既定で832×480、81 framesです。48GBでは5秒セクションを順番に実行し、RIFE/upscaleを最後に有効化します。

## 4. 初回起動

Logsで次を確認します。

```text
MODEL PROFILE: lightning-longvideo (14 assets)
TRANSFER ENGINE: 4 files in parallel
...
[check_env] profile=lightning-longvideo assets=14 missing=0
```

初回は約72.39 GBです。途中でPodを停止しても、同じVolumeならHF cacheまたは`.part`から再開します。`missing=0`になった後、ConnectからHTTP Service Port 8188を開きます。

## 5. workflowを開く

Workflowsから`wan22_native_enhanced_lightning_longvideo_runpod`を開きます。最初は5秒だけ生成し、結果が良ければ10秒、15秒、20秒のgroupを順に有効化してください。

## 6. 更新

新しいGitHub Actions buildが成功した後にPodを作り直します。`/workspace`のモデル、input、output、ユーザーworkflowはVolumeに残ります。bundle workflowが更新された場合も既存版を上書きせず、hash付きの新しい版が追加されます。
