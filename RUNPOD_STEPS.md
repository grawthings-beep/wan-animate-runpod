# RunPod Template設定

## 1. Imageを選ぶ

GitHub Actionsの最新成功runに対応するcommit SHAを使います。

4090:

```text
ghcr.io/grawthings-beep/wan-animate-runpod:loop-ada-cu128-sha-<40文字SHA>
```

5090 / RTX PRO Blackwell:

```text
ghcr.io/grawthings-beep/wan-animate-runpod:loop-blackwell-cu130-sha-<40文字SHA>
```

`loop-ada-cu128`と`loop-blackwell-cu130`は便利な可変tagですが、RunPod側の古いcacheを避けるため本番はSHA tag推奨です。

## 2. Storage

- Container Disk: 20 GB以上
- Volume Disk: `loop-core`は80 GB以上、`loop-all`は100 GB推奨
- Volume mount path: `/workspace`
- Network Volume: なしでよい

Volume Diskは生成物、model、HF cacheを保持します。同じPodのStop/Startでは再利用できますが、Terminate後の新Podには引き継がれません。

## 3. Networking configuration

HTTP Portsへ1行追加します。

```text
Port label: ComfyUI
Port number: 8188
```

TCP Portsは空で構いません。`PORT=8188`と`LISTEN=0.0.0.0`も環境変数に残します。前者はRunPod proxyの公開先、後者はcontainer内でComfyUIと起動status serverが待ち受ける設定なので役割が別です。

## 4. Secrets

RunPod Secretsに次を作ります。

- `HF_TOKEN`: Hugging Face read token
- `CIVITAI_TOKEN`: CivitAI API token

Environment variables欄では値を平文で貼らず、鍵アイコンからSecretを割り当てます。

## 5. Environment variables（コピペ用）

```text
PORT=8188
LISTEN=0.0.0.0
DOWNLOAD_MODELS=1
MODEL_PROFILE=loop-all
RUN_DEP_CHECK=1
BOOTSTRAP_STATUS=1
BOOT_FAILURE_HOLD_SECONDS=900
DOWNLOAD_WORKERS=4
ARIA2_CONNECTIONS=16
ARIA2_SPLITS=16
HF_SNAPSHOT_WORKERS=8
HF_XET_HIGH_PERFORMANCE=1
HF_XET_NUM_CONCURRENT_RANGE_GETS=64
HF_XET_CHUNK_CACHE_SIZE_BYTES=0
HF_HUB_DOWNLOAD_TIMEOUT=300
CUDA_NORMALIZE_VISIBLE_DEVICES=1
CUDA_PREFLIGHT=1
CUDA_READY_TIMEOUT=90
CUDA_READY_INTERVAL=10
MODEL_DISK_PREFLIGHT=1
MODEL_DISK_HEADROOM_GB=12
YOLO_CONFIG_DIR=/workspace/.cache/ultralytics
YOLO_AUTOINSTALL=false
YOLO_OFFLINE=true
COMFYUI_ARGS=--reserve-vram 3 --max-upload-size 300
CIVITAI_API_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
```

LoRAをONの6行だけに絞るなら、次の1行だけ変更します。

```text
MODEL_PROFILE=loop-core
```

その場合は名前に`core`が付くworkflowを開いてください。profileとworkflowを混ぜると、意図的に省いたOFF LoRAをComfyUIが不足扱いします。

## 6. 起動確認

Deploy後すぐにConnectの8188を開きます。最初は起動status pageになり、次の順で表示が進みます。

```text
cuda-preflight -> workflows -> models -> validation -> ready -> ComfyUI
```

正常ログの要点:

```text
[gpu-preflight] TORCH STACK READY
[gpu-preflight] READY
MODEL PROFILE: loop-all (26 assets)
TRANSFER ENGINE: 4 files in parallel
[check_env] ... required_missing=0
BOOT PHASE: comfyui-exec
```

4090 imageを5090で使った場合、または逆の場合は`wrong image for GPU`でdownload前に止まります。5090 hostのdriverが580未満なら`incompatible CUDA 13 driver`です。この場合はStop/StartではなくPodをTerminateし、正しいimageで新規Deployしてください。

Edgeだけ403になりChromeでは開く場合、RunPod proxy自体ではなくEdge側に残ったRunPod認証cookie・追跡防止・拡張機能が原因です。InPrivateで開く、`runpod.net`のsite dataを削除、追跡防止をBalancedへ変更の順で確認します。

## Batch10の一括投入

名前に`batch10`が付くworkflowを開き、`BULK DROP + QUEUE 10 LOOPS` nodeを使います。

1. `01.png`〜`10.png`を入れたフォルダを一括投入欄へdropします。10枚入りZIPでも構いません。
2. 対応するpositive promptを1行ずつ書いた`prompts.txt`を同じ欄へdropします。
3. `準備完了: 画像10枚 + prompt 10件`を確認します。
4. `QUEUE 10 LOOPS (SEQUENTIAL)`を1回だけ押します。

フォルダ名やOSの列挙順ではなく、相対ファイル名の自然順で対応付けます。たとえば`2.png`は`10.png`より前です。ZIP内に`prompts.txt`も入れた場合は、ZIP 1個をdropするだけで1〜3が完了します。画像またはpromptが10件ちょうどでなければqueueされません。
