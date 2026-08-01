# WAN 2.2 ComfyUI on RunPod

RunPodで次の2系統をそのまま開けるComfyUIイメージです。

- Smooth Workflow v6.0: I2V / T2V / First-to-Last Frame / MMAudio / シームレスループ
- Native Enhanced Lightning Long Video: 5〜20秒の連続I2V、RIFE 60fps化、2倍アップスケール

ワークフロー、18個のcustom-node pack、モデル/LoRA manifestを同じcommitで固定しています。Network Volumeは不要です。新しいPodではローカルVolume Diskへモデルを高速取得し、同じPodをStop/Startした場合だけ検証済みファイルと途中ダウンロードを再利用します。PodをTerminateすればVolume Diskも削除され、次のPodでは最初から取得します。

## コンテナイメージ

`main`へのpushでGitHub Actionsが次を公開します。

```text
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-lightning-longvideo
ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
ghcr.io/grawthings-beep/wan-animate-runpod:latest
```

RunPodから認証なしでpullする場合、GitHub PackagesでパッケージをPublicにしてください。

## シームレスループのRunPod設定

```text
Template type: Pod
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
Container disk: 50 GB
Volume disk: 100 GB以上（生成動画を多く残すなら150 GB推奨）
Volume mount path: /workspace
HTTP port: 8188
```

環境変数:

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
HF_XET_CHUNK_CACHE_SIZE_BYTES=0
HF_HUB_DOWNLOAD_TIMEOUT=300
CUDA_PREFLIGHT=1
CUDA_READY_TIMEOUT=90
CUDA_READY_INTERVAL=10
MODEL_DISK_PREFLIGHT=1
MODEL_DISK_HEADROOM_GB=12
COMFYUI_ARGS=--reserve-vram 3
```

`CIVITAI_API_TOKEN`と`HF_TOKEN`は平文入力ではなく、RunPod Secretsの鍵アイコンから割り当てます。既定の`loop-quality`はHugging Faceだけで完結するため、CivitAI tokenなしでも起動できます。HTTP portはNetworking configurationに`8188`を追加し、`PORT`/`LISTEN`は上記の環境変数にも残します。

## 高速ダウンロード

既定の`loop-quality`は11 assets、約40.94 GBです。4本をサイズ順に並列取得します。

- Hugging Face: Rust製`hf_xet`、64 range requests/file、同一Volume上のcacheからhard-linkでzero-copy
- Xet chunk cache: 一回限りの新規取得には不利なので`0`。公式既定と同じく余分な最大10 GBを使わない
- CivitAI: 実URL解決後にaria2の16分割転送
- 再開: HF cacheとaria2の`.part`を永続Volumeに保持
- 完全性: 公開元のsizeとSHA256を全単一ファイルで検証

Network Volumeを使わなくても、RunPodのローカルVolume Diskは`/workspace`へマウントされます。同じPodのStop/Startでは保持され、PodのTerminateで削除されます。毎回ダウンロードする方針なら、PodをTerminateして新規デプロイしてください。429や帯域制限が出る環境だけ`DOWNLOAD_WORKERS=2`へ下げてください。

ダウンロード前に、選択profileの未取得容量と12 GBの作業・出力余白を確認します。`loop-quality`では約52.94 GB以上の空きが必要です。テンプレートはVolume Disk 100 GBにしてください。容量不足なら1 byteも取得する前に停止し、`Disk quota exceeded`を防ぎます。

## モデルprofile

| `MODEL_PROFILE` | Assets | 容量 | 用途 |
|---|---:|---:|---|
| `lightning-longvideo` | 12 | 43.80 GB | 今回のNative Enhanced Lightning。接続済みQ8 High/Low、全候補LoRA、RIFE、upscaler |
| `i2v-quality` | 8 | 39.12 GB | Smooth v6のI2V |
| `loop-quality` | 11 | 40.94 GB | Smooth v6の音なしシームレスループ＋NSFW-22＋SmoothXXXAnimation High/Low |
| `t2v-quality` | 7 | 40.68 GB | Smooth v6のT2V |
| `full` | 38 | 145.44 GB | 両ワークフローの全asset |

`loop-quality`が既定です。First-to-Last Frameの実行経路だけを残し、LightX2V rank128、NSFW-22 High/Low、SmoothXXXAnimation High/Lowを取得します。別ブランチのLoRA、未接続GGUF、MMAudioはダウンロードもworkflow表示も行いません。`lightning-longvideo`を明示した場合だけNative Enhanced Lightning用assetを取得します。

RunPodは同じ可変Dockerタグをキャッシュする場合があります。更新直後のPodで古いworkflowや不足モデルが表示された場合は、既存PodのStop/Startではなく新規Podを作成し、Actionsが発行する`sha-<40文字のcommit SHA>`タグをContainer Imageに指定してください。起動ログの`BUNDLE REVISION`がそのSHAと一致し、`[check_env] ... required_missing=0`になるまでComfyUIは起動しません。

ループworkflowのLoRA:

- `LightX2V rank128`: 少ないstep数で生成するためのアクセラレータ。High 3.0 / Low 1.5で既定ON
- `NSFW-22-H/L-e8`: CubeyAIのWAN 2.2 General NSFW v0.08a。High 2.0 / Low 1.0で既定ON
- `SmoothXXXAnimation High/Low`: SmoothMix用animation LoRA。High/Lowとも0.5で追加済みだが既定OFF

NSFW-22は指定どおりHigh 2.0 / Low 1.0に固定しています。LightX2V High 3.0 / Low 1.5と同時にONです。SmoothXXXAnimationもONにすると重ね掛けがさらに強くなるため、まずHigh/Lowとも0.25〜0.5で試してください。4本はprivate Hugging Face backupから取得するため`HF_TOKEN`が必要です。

## ワークフロー

初回起動後、ComfyUIのWorkflowsから選べます。

- `wan22_native_enhanced_lightning_longvideo_runpod`
- `wan22_smooth_v6_aio_runpod`
- `wan22_smooth_v6_seamless_loop_runpod`

ループ版の流れ:

1. `wan22_smooth_v6_seamless_loop_runpod`を開く。
2. FIRST FRAMEとLAST FRAMEへ同じ画像を入れる。
3. まず81 framesで、往復ではなく一周する連続動作をpromptへ書く。
4. 短いループの継ぎ目を確認してから長さやRIFE補間を増やす。

## GPU

- 推奨: H100 80GB / A100 80GB
- 実用: L40S / RTX 6000 Ada 48GB（832×480、81 framesを基準）
- RTX 5090 32GB: CUDA 12.8 imageで対応。短尺の確認用。ホスト品質にばらつきがあるため起動時検査あり
- RTX 4090 24GB: 低解像度・短尺の確認用。長尺・RIFE・upscale用途には非推奨

最初から長くせず、81 framesで継ぎ目を確認してください。RIFEは生成結果が安定してから有効化する方がVRAMと時間を無駄にしません。

### 4090/5090の`CUDA unknown error`

RunPodではPodが特定の物理ホストへ割り当てられます。問題のログでは5090を`nvidia-smi`は認識している一方、CUDA 12.8版PyTorchの`torch.cuda.is_available()`が`False`で、実テンソル演算も`CUDA unknown error`になっていました。これはVRAM不足やworkflow設定ではなく、その割り当てホストのCUDA初期化失敗です。

起動時に別Pythonプロセスで`nvidia-smi`と実CUDA演算を最大90秒検査します。合格するまでモデルダウンロードは始めません。次が出た場合はPodを**Terminateして新規デプロイ**してください。Stop/Startは同じ物理ホストに紐づいたままになる場合があります。

```text
[gpu-preflight] FATAL: this Pod's assigned GPU cannot execute CUDA.
No model download was started.
```

`CUDA out of memory`は別問題です。その場合は832×480・81 framesから始め、RIFEと2倍upscaleを切って確認してください。

## 永続化レイアウト

```text
/workspace/
├── comfyui/
│   ├── models/
│   ├── input/
│   ├── output/
│   └── user/default/workflows/
├── config/wan22-models.json
└── .cache/
```

## ローカル検証

```bash
python scripts/prepare_workflows.py --check
python scripts/validate_assets.py
python -m unittest discover -s tests -v
bash -n scripts/common.sh scripts/install_custom_nodes.sh scripts/start.sh
```

詳しいRunPod画面の入力値は[RUNPOD_STEPS.md](RUNPOD_STEPS.md)、全環境変数は[runpod-template.env.example](runpod-template.env.example)を参照してください。
