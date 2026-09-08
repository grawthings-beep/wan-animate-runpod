# RunPod Template設定

## 1. Imageを選ぶ

GitHub Actionsの最新成功runに対応するcommit SHAを使います。

4090 / 5090共通の第一候補:

```text
ghcr.io/grawthings-beep/wan-animate-runpod:loop-cu128-sha-<40文字SHA>
```

CUDA13対応driver環境の代替（5090専用ではありません）:

```text
ghcr.io/grawthings-beep/wan-animate-runpod:loop-cu130-sha-<40文字SHA>
```

`loop-cu128`と`loop-cu130`は可変tagなので本番はSHA tag推奨です。旧`loop-ada-cu128` / `loop-blackwell-cu130`名もaliasとして発行します。GPU名だけの拒否は廃止しました。CI成功はbuild/CPU検証であり、4090/5090の実機生成を保証しません。

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

追加LoRAの選択肢をNSFW-22とSmoothXXXAnimationの2組＋wind（初期値OFF）に絞るなら、次の1行だけ変更します。

```text
MODEL_PROFILE=loop-core
```

workflow名はどちらのprofileでも同じ2本です。旧名の`core` workflowを探す必要はありません。

以前の設定からは`CUDA_NORMALIZE_VISIBLE_DEVICES`を削除してください。残っていても新imageでは無視します。`CUDA_VISIBLE_DEVICES`自体は通常追加不要で、RunPodが与えた値を保持します。空の値や`-1`を自分で設定するとGPUが非表示になるため、その場合は診断に`device-selection`と表示します。

## 6. 起動確認

Deploy後すぐにConnectの8188を開きます。最初は起動status pageになり、次の順で表示が進みます。

```text
cuda-preflight -> workflows -> models -> validation -> ready -> ComfyUI
```

正常ログの要点:

```text
[gpu-preflight] TORCH STACK READY
[gpu-preflight] READY
MODEL PROFILE: loop-all (30 assets)
TRANSFER ENGINE: 4 files in parallel
[check_env] ... required_missing=0
BOOT PHASE: comfyui-exec
```

失敗した場合は8188の「診断ファイルを保存」を押してください。iPhoneなら保存した`wan-gpu-diagnostics.json`をそのまま送れます。端末コマンド不要です。ファイルにはGPU/Pod IDなどが含まれるため公開しないでください。起動チェック中の8188にのみdownloadボタンがあり、ComfyUIへの切り替え後も元のJSONは`/workspace/config/gpu-diagnostics.json`に残ります。

| 表示 | 確認対象 |
|---|---|
| `runtime-stack` | PyTorch依存関係の読み込み・固定版との一致 |
| `device-selection` | GPUを非表示にするCUDA_VISIBLE_DEVICES指定 |
| `device-access` | /dev/nvidia*の公開・権限・UVMのI/Oエラー |
| `driver-library` / `driver-compatibility` | libcuda・CUDAとdriverの互換性 |
| `cuda-memory` / `cuda-operation` | 最小CUDA演算のメモリ不足・実行失敗 |

GPUの故障と断定したり、一律にPod削除を求めたりしません。モデルはCUDA確認成功まで取得しません。失敗ページは既定900秒保持し、Podの停止・削除は自動で行いません。停止中ではなく起動失敗で待機中のPodには料金が発生する場合があります。生成物や診断を保存せずにTerminateしないでください。

Edgeだけ403になりChromeでは開く場合、RunPod proxy自体ではなくEdge側に残ったRunPod認証cookie・追跡防止・拡張機能が原因です。InPrivateで開く、`runpod.net`のsite dataを削除、追跡防止をBalancedへ変更の順で確認します。

## 開くワークフローは2本だけ

- 単体：`wan22_loop_single_runpod.json`
- 10本逐次：`wan22_loop_batch10_runpod.json`

両方ともAIアップスケール・RIFE・自動モザイク付き。core/通常/モザイク有無の別ファイルは廃止しました。`MODEL_PROFILE=loop-core`でも同じ2本を使い、未取得LoRAの行だけを起動時に除きます。環境変数の追加はありません。

旧同梱11本・hash付きの旧コピー・更新前の同梱名の編集版は`/workspace/comfyui/workflow-backups/`へ退避します。任意の名前で保存したユーザーworkflowはそのままです。新しいimageを使っても開きっぱなしのcanvasは旧版のため、必ず新しい名前のworkflowを開いてください。

単体ではFIRST/LASTへ同じ画像を選び、positive promptを入力して通常のQueueを押します。初期生成サイズ720×960、最終出力1440×1920、約5秒。自動モザイクの対象と輪郭は従来どおりです。

## Enhanced V2 Q8の初回確認

新しいSHA imageを使い、同梱workflowを開き直します。モデル欄が`ENHANCED V2 Q8`のHigh/Low、samplerが両方`KSamplerAdvanced`、全LoRAがOFFなら新presetです。古いJSONのモデル名だけを変えるとGGUFを読み込めません。

設定は両samplerで`steps=4 / cfg=1 / euler / simple`、High `start=0 / end=2`、Low `start=2 / end=4`。LightX2V/Lightningは追加しません。CFG 1ではnegative promptは無効です。batch10では同一画像が両端へ自動で配線されます。

アップスケールの選択欄は`2xNomosUni_span_multijpg.safetensors`になっていることを確認します。「4」しか表示される場合は旧workflowです。旧NMKDを選んでも最終倍率2倍は維持されますが重くなります。RIFEは拡大前に処理し、ensemble OFF、batch_size=1。その後SPANで2倍にします。モザイクのdeviceはautoでGPU優先・OOM時CPUへ復帰し、cpu固定も選べます。全フレームを検出する点は変わりません。

生成が遅い場合は`[wan-post]`のupscale/mosaic秒数と、sampler進捗・`Prompt executed in ...`を含むログで比較します。初回ロード込みと2回目では時間が違うため、GPU名と何回目の生成かも控えてください。

## Wind motion LoRAの使用

`wind.safetensors`は両方のworkflowのLow側に1.0 / OFFで残しています。相性や最適強度は未確認なので、まずOFFで比較します。batch10ではLoRA設定が10本に共通で適用されます。

## Batch10の一括投入

名前に`batch10`が付くworkflowを開き、`BULK DROP + QUEUE 10 LOOPS` nodeを使います。

1. `01.png`〜`10.png`を入れたフォルダを一括投入欄へdropします。10枚入りZIPでも構いません。
2. 対応するpositive promptを空行で10ブロックに区切った`prompts.txt`を同じ欄へdropします。各promptブロック内は複数行で構いません。
3. `準備完了: 画像10枚 + prompt 10件`を確認します。
4. `QUEUE 10 LOOPS (SEQUENTIAL)`を1回だけ押します。

フォルダ名やOSの列挙順ではなく、相対ファイル名の自然順で対応付けます。たとえば`2.png`は`10.png`より前です。ZIP内に`prompts.txt`も入れた場合は、ZIP 1個をdropするだけで1〜3が完了します。画像またはpromptが10件ちょうどでなければqueueされません。

`prompts.txt`の例:

```text
subject one performs a continuous cyclic motion,
medium shot, fixed camera, stable background

subject two turns slowly and returns to the starting pose,
full body shot, smooth periodic motion

（同じ形式で合計10ブロック）
```
