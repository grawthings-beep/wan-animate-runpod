# WAN 2.2 ComfyUI on RunPod

## Fast loop image

ループ専用RunPod Templateでは、Actionsが発行する次のimmutable tagを使います。

```text
ghcr.io/grawthings-beep/wan-animate-runpod:minimal-sha-<40文字のcommit SHA>
```

このimageは公式PyTorch 2.10.0/cu128 runtimeと4種類のseamless-loop workflowに必要な
custom nodeだけを含みます。Jupyter、SSH、FileBrowser、Manager、MMAudio、GGUF、
長尺動画系を焼かないため、旧全入りimageの5.866 GiBから5.523 GiBへ縮小しています。
`wan22-loop-minimal`は同内容の便利な可変tagですが、RunPodの古いimage cacheを避ける
ため、本番では`minimal-sha-*`を推奨します。AIO / Native Lightning / MMAudioを使う
場合だけ従来の`sha-<commit SHA>`を使います。

起動ログの次の3行で、container process開始後にCUDA確認とモデル取得がそれぞれ何秒
かかったかを分離できます。image pull時間はcontainer開始前なのでRunPod画面側で確認します。

```text
BOOT PHASE: cuda-preflight-complete elapsed=...s
BOOT PHASE: model-download-complete elapsed=...s
BOOT PHASE: comfyui-exec elapsed=...s
```

## 10本を順番に生成してZIPを自動ダウンロード

`wan22_smooth_v6_seamless_loop_batch10_runpod` は、10枚の画像と10個の対応するpositive promptを登録できる派生ワークフローです。専用の `QUEUE 10 LOOPS (SEQUENTIAL)` ボタンを1回押すと、通常のループ生成を独立した10件のComfyUIジョブとして順番に登録します。同時バッチではないため、1本が終了してVRAMを解放してから次の1本へ進みます。

10本目まで成功すると、`slot-01.mp4` から `slot-10.mp4` と `manifest.json` を含むZIPをブラウザが自動ダウンロードします。途中のジョブが失敗した場合は、不完全なZIPを誤ってダウンロードしません。サーバー側にも `/workspace/comfyui/output/Video/loop-batches/<batch-id>/` が残ります。

使い方は [BATCH10_WORKFLOW.md](BATCH10_WORKFLOW.md) を参照してください。

RunPodで次の2系統をそのまま開けるComfyUIイメージです。

- Smooth Workflow v6.0: I2V / T2V / First-to-Last Frame / MMAudio / シームレスループ
- Native Enhanced Lightning Long Video: 5〜20秒の連続I2V、RIFE 60fps化、2倍アップスケール

ワークフロー、18個の外部custom-node pack、1個の同梱batchノード、モデル/LoRA manifestを同じcommitで固定しています。Network Volumeは不要です。新しいPodではローカルVolume Diskへモデルを高速取得し、同じPodをStop/Startした場合だけ検証済みファイルと途中ダウンロードを再利用します。PodをTerminateすればVolume Diskも削除され、次のPodでは最初から取得します。

## コンテナイメージ

`main`へのpushでGitHub Actionsが次を公開します。

```text
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-lightning-longvideo
ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
ghcr.io/grawthings-beep/wan-animate-runpod:latest
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-loop-minimal
```

RunPodから認証なしでpullする場合、GitHub PackagesでパッケージをPublicにしてください。

## シームレスループのRunPod設定

```text
Template type: Pod
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:minimal-sha-<40文字のcommit SHA>
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
CUDA_NORMALIZE_VISIBLE_DEVICES=1
CUDA_PREFLIGHT=1
CUDA_READY_TIMEOUT=90
CUDA_READY_INTERVAL=10
MODEL_DISK_PREFLIGHT=1
MODEL_DISK_HEADROOM_GB=12
YOLO_CONFIG_DIR=/workspace/.cache/ultralytics
YOLO_AUTOINSTALL=false
YOLO_OFFLINE=true
COMFYUI_ARGS=--reserve-vram 3
```

`CIVITAI_API_TOKEN`と`HF_TOKEN`は平文入力ではなく、RunPod Secretsの鍵アイコンから割り当てます。既定の`loop-quality`は輪郭モザイク用segmentation modelをCivitAIから取得するため、両方のtokenが必要です。HTTP portはNetworking configurationに`8188`を追加し、`PORT`/`LISTEN`は上記の環境変数にも残します。

## 高速ダウンロード

既定の`loop-quality`は26 assets、約46.13 GBです。4本をサイズ順に並列取得します。自動モザイク用segmentation modelと、任意のJOI Handjob Trend High/Low LoRAも含みます。

- Hugging Face: Rust製`hf_xet`、64 range requests/file、同一Volume上のcacheからhard-linkでzero-copy
- Xet chunk cache: 一回限りの新規取得には不利なので`0`。公式既定と同じく余分な最大10 GBを使わない
- CivitAI: 実URL解決後にaria2の16分割転送
- 再開: HF cacheとaria2の`.part`を永続Volumeに保持
- 完全性: 公開元のsizeとSHA256を全単一ファイルで検証

Network Volumeを使わなくても、RunPodのローカルVolume Diskは`/workspace`へマウントされます。同じPodのStop/Startでは保持され、PodのTerminateで削除されます。毎回ダウンロードする方針なら、PodをTerminateして新規デプロイしてください。429や帯域制限が出る環境だけ`DOWNLOAD_WORKERS=2`へ下げてください。

ダウンロード前に、選択profileの未取得容量と12 GBの作業・出力余白を確認します。`loop-quality`では約58.13 GB以上の空きが必要です。テンプレートはVolume Disk 100 GBにしてください。容量不足なら1 byteも取得する前に停止し、`Disk quota exceeded`を防ぎます。

## モデルprofile

| `MODEL_PROFILE` | Assets | 容量 | 用途 |
|---|---:|---:|---|
| `lightning-longvideo` | 12 | 43.80 GB | 今回のNative Enhanced Lightning。接続済みQ8 High/Low、全候補LoRA、RIFE、upscaler |
| `i2v-quality` | 8 | 39.12 GB | Smooth v6のI2V |
| `loop-quality` | 26 | 46.13 GB | Smooth v6の音なしシームレスループ＋各LoRA＋RIFE＋CPU輪郭モザイク |
| `t2v-quality` | 7 | 40.68 GB | Smooth v6のT2V |
| `full` | 53 | 150.63 GB | 全workflowの全asset |

`loop-quality`が既定です。First-to-Last Frameの実行経路だけを残し、LightX2V rank128、NSFW-22 High/Low、SmoothXXXAnimation High/Low、Anime Cumshot Aesthetics High/Low、JOI Handjob Trend High/Low、iroiroLoRA High/Low 5組を取得します。別ブランチのLoRA、未接続GGUF、MMAudioはダウンロードもworkflow表示も行いません。`lightning-longvideo`を明示した場合だけNative Enhanced Lightning用assetを取得します。

RunPodは同じ可変Dockerタグをキャッシュする場合があります。更新直後のPodで古いworkflowや不足モデルが表示された場合は、既存PodのStop/Startではなく新規Podを作成し、ループ用ならActionsが発行する`minimal-sha-<40文字のcommit SHA>`タグをContainer Imageに指定してください。起動ログの`BUNDLE REVISION`がそのSHAと一致し、`[check_env] ... required_missing=0`になるまでComfyUIは起動しません。

ループworkflowのLoRA:

- `LightX2V rank128`: 少ないstep数で生成するためのアクセラレータ。High 3.0 / Low 1.5で既定ON
- `NSFW-22-H/L-e8`: CubeyAIのWAN 2.2 General NSFW v0.08a。High 2.75 / Low 1.65で既定ON
- `SmoothXXXAnimation High/Low`: SmoothMix用animation LoRA。High 1.5 / Low 1.0で既定ON
- `Anime Cumshot Aesthetics High/Low`: 作者推奨のHigh 1.0 / Low 1.0で追加。公式WANベース向けのため既定OFF
- `JOI Handjob Trend High/Low`: WAN 2.2 I2V-A14B専用。High 1.0 / Low 1.0で既定OFF。使用時は両方をON
- `cheek_bulge_fellatio`: WAN 2.2 High＋WAN 2.1 Low。triggerは`cheek bulge, fellatio`。各1.0で既定OFF
- `glans_licking`: WAN 2.2 High＋WAN 2.1 Low。triggerは`glans licking`。各1.0で既定OFF
- `head_back`: WAN 2.2 High＋WAN 2.1 Low。triggerは`head back`。各1.0で既定OFF
- `paizuri_unaligned_breasts`: WAN 2.2 High＋WAN 2.1 Low。triggerは`paizuri, unaligned breasts`。各1.0で既定OFF
- `washizukami`: WAN 2.2 High＋WAN 2.1 Low。作者記載のtriggerは`grabbin own/another's breast, deep skin`。各1.0で既定OFF

指定どおりLightX2V High 3.0 / Low 1.5、NSFW-22 High 2.75 / Low 1.65、SmoothXXXAnimation High 1.5 / Low 1.0を同時にONにし、既定解像度を528×704に固定しています。Anime Cumshot AestheticsはSmoothMixとの組み合わせをユーザーが明示的に試す場合だけHigh/LowをONにしてください。JOI Handjob TrendはHigh/Lowを1.0で両方ONにし、舌を出す動作と手を素早く上下へ反復させる動作をpromptへ明記します。iroiroLoRAは目的に合うHigh/Lowの1組だけを両方ONにするのが基本です。private Hugging Face backupの6本には`HF_TOKEN`、JOI pairと輪郭モザイクmodelには`CIVITAI_API_TOKEN`が必要です。

## ワークフロー

初回起動後、ComfyUIのWorkflowsから選べます。

- `wan22_native_enhanced_lightning_longvideo_runpod`
- `wan22_smooth_v6_aio_runpod`
- `wan22_smooth_v6_seamless_loop_runpod`
- `wan22_smooth_v6_seamless_loop_batch10_runpod`
- `wan22_smooth_v6_seamless_loop_auto_mosaic_runpod`
- `wan22_smooth_v6_seamless_loop_batch10_auto_mosaic_runpod`

ループ版の流れ:

1. `wan22_smooth_v6_seamless_loop_runpod`を開く。
2. FIRST FRAMEとLAST FRAMEへ同じ画像を入れる。
3. まず81 framesで、往復ではなく一周する連続動作をpromptへ書く。
4. 短いループの継ぎ目を確認してから長さやRIFE補間を増やす。

## 完成動画への自動モザイク

既存workflowは変更せず、末尾が`_auto_mosaic_runpod`の2本を別に追加しています。どちらも`RIFE VFI → WAN Auto Mosaic JUST Segmentation (CPU) → VHS Video Combine`の順です。入力画像には処理せず、補間済みの完成フレームへモザイクを入れてから1回だけMP4エンコードします。

- `coverage_preset=JUST`: 検出boxではなく専用YOLO11-segの輪郭maskを使い、境界を4%だけ膨張。iPhone版AutoMosaicのJUSTと同じ設計
- `confidence=0.30` / `iou_threshold=0.50`: iPhone版と同じ初期値。誤検出が多ければconfidenceを0.40〜0.55へ上げる
- `block_size=0`: 自動（短辺÷50、最低10 px）。固定したい場合だけ24〜36などを指定
- `max_gap_frames=3`: 検出に成功したframeはそのframe自身の輪郭だけを使用。一瞬の未検出だけを補間し、ループ境界も循環処理
- `target_classes=pussy,anus,penis,testicles`: 専用modelの対象。必要な場合だけ`nipples`を追加

`WIDE`と`SAFE`は輪郭maskへ大きな楕円を足す安全側presetなので、ジャストに掛けたい場合は`JUST`のまま使います。従来の「矩形を18%拡大して前後frameへunion」は廃止しました。検出はCPU固定なのでWAN/RIFEのVRAMを消費しません。10本版も従来どおり1ジョブずつ生成し、10本目の終了後にモザイク済みMP4をZIPでダウンロードします。自動検出には見逃しの可能性があるため、公開前には必ず完成動画を目視確認してください。検出モデルは[Anime NSFW Detection / ADetailer All-in-One v5.0](https://civitai.com/models/1313556)（YOLO11 segmentation）、推論runtimeは[Ultralytics](https://pypi.org/project/ultralytics/) 8.4.104（AGPL-3.0）へ固定しています。

## GPU

- 推奨: H100 80GB / A100 80GB
- 実用: L40S / RTX 6000 Ada 48GB（832×480、81 framesを基準）
- RTX 5090 32GB: CUDA 12.8 imageで対応。短尺の確認用。ホスト品質にばらつきがあるため起動時検査あり
- RTX 4090 24GB: 低解像度・短尺の確認用。長尺・RIFE・upscale用途には非推奨

最初から長くせず、81 framesで継ぎ目を確認してください。RIFEは生成結果が安定してから有効化する方がVRAMと時間を無駄にしません。

### 4090/5090の`CUDA unknown error`

RunPodではPodが特定の物理ホストへ割り当てられます。問題のログでは5090を`nvidia-smi`は認識している一方、CUDA 12.8版PyTorchの`torch.cuda.is_available()`が`False`で、実テンソル演算も`CUDA unknown error`になっていました。これはVRAM不足やworkflow設定ではなく、その割り当てホストのCUDA初期化失敗です。

起動時、`nvidia-smi`がGPUを1台だけ返す場合はTorchをimportする前に`CUDA_VISIBLE_DEVICES=0`へ正規化します。続いて、固定済みのTorch 2.10.0 / TorchVision 0.25.0 / TorchAudio 2.10.0 cu128が混在していないことと、別Pythonプロセスでの実CUDA演算を検査します。ベースイメージはDocker Hubタグだけでなくlinux/amd64 manifest digestまで固定しています。

CUDA初期化は最大90秒待ちますが、RTX 5090/BlackwellでNVIDIA driverが570.26未満なら再試行しても動かないため即停止します。合格するまでモデルダウンロードは始めません。次が出た場合はPodを**Terminateして新規デプロイ**してください。Stop/Startは同じ物理ホストに紐づいたままになる場合があります。

```text
[gpu-preflight] FATAL: this Pod's assigned GPU cannot execute CUDA.
No model download was started.
```

正常時はダウンロードより前に次の3種類が出ます。

```text
[cuda-bootstrap] Using CUDA_VISIBLE_DEVICES=0 for the single GPU exposed by RunPod (...)
[gpu-preflight] TORCH STACK READY {"torch": "2.10.0+cu128", ...}
[gpu-preflight] READY attempt=1
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
