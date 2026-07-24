# WAN 2.2 ComfyUI on RunPod

RunPodで次の2系統をそのまま開けるComfyUIイメージです。

- Smooth Workflow v6.0: I2V / T2V / First-to-Last Frame / MMAudio / シームレスループ
- Native Enhanced Lightning Long Video: 5〜20秒の連続I2V、RIFE 60fps化、2倍アップスケール

ワークフロー、18個のcustom-node pack、モデル/LoRA manifestを同じcommitで固定しています。初回起動時だけ永続Volumeへモデルを取得し、以後は検証済みファイルと途中ダウンロードを再利用します。

## コンテナイメージ

`main`へのpushでGitHub Actionsが次を公開します。

```text
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-lightning-longvideo
ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
ghcr.io/grawthings-beep/wan-animate-runpod:latest
```

RunPodから認証なしでpullする場合、GitHub PackagesでパッケージをPublicにしてください。

## Lightning長尺ワークフローのRunPod設定

```text
Template type: Pod
Container image: ghcr.io/grawthings-beep/wan-animate-runpod:wan22-lightning-longvideo
Container disk: 50 GB
Volume disk: 200 GB以上（全profileを残すなら300 GB推奨）
Volume mount path: /workspace
HTTP port: 8188
```

環境変数:

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

`CIVITAI_API_TOKEN`と`HF_TOKEN`は平文入力ではなく、RunPod Secretsの鍵アイコンから割り当てます。HTTP portはNetworking configurationに`8188`を追加し、`PORT`/`LISTEN`は上記の環境変数にも残します。

## 高速ダウンロード

`lightning-longvideo`は14 assets、約72.39 GB（67.42 GiB）です。4本をサイズ順に並列取得します。

- Hugging Face: Rust製`hf_xet`、64 range requests/file、同一Volume上のcacheからhard-linkでzero-copy
- CivitAI: 実URL解決後にaria2の16分割転送
- 再開: HF cacheとaria2の`.part`を永続Volumeに保持
- 完全性: 公開元のsizeとSHA256を全単一ファイルで検証

Podを止めても`/workspace`を残せば、次回は完成済みファイルを再取得しません。429や帯域制限が出る環境だけ`DOWNLOAD_WORKERS=2`へ下げてください。

## モデルprofile

| `MODEL_PROFILE` | Assets | 容量 | 用途 |
|---|---:|---:|---|
| `lightning-longvideo` | 14 | 72.39 GB | 今回のNative Enhanced Lightning。FP8/Q8両方、全候補LoRA、RIFE、upscaler |
| `i2v-quality` | 8 | 39.12 GB | Smooth v6のI2V |
| `loop-quality` | 26 | 64.70 GB | Smooth v6のシームレスループ |
| `t2v-quality` | 7 | 40.68 GB | Smooth v6のT2V |
| `full` | 36 | 144.22 GB | 両ワークフローの全asset |

LightningワークフローのQ8 High/Lowが既定の実行経路です。FP8 High/Lowも代替用として同じprofileで取得します。Enhanced V2 checkpointにはLightningが焼き込み済みなので、ワークフローへ表示した追加LoRAはすべてOFFにしてあります。

## ワークフロー

初回起動後、ComfyUIのWorkflowsから選べます。

- `wan22_native_enhanced_lightning_longvideo_runpod`
- `wan22_smooth_v6_aio_runpod`
- `wan22_smooth_v6_seamless_loop_runpod`

Lightning版の流れ:

1. Load Imageを自分の開始画像に差し替える。
2. まず最初の5秒セクションだけを生成し、seedとpromptを固める。
3. 気に入ったら次の10/15/20秒セクションを順に有効化する。
4. 最後にRIFE 60fpsまたはupscale groupを必要なときだけ有効化する。

各セクションの`initial_reference_image`には最初の画像が渡され、`previous_video`の末尾フレームから続きます。`motion_amplitude=1.15`を基準にし、遅ければ1.2〜1.3へ上げます。色ずれは`color_protect=True`、`correct_strength=0.01〜0.05`から調整します。

## GPU

- 推奨: H100 80GB / A100 80GB
- 実用: L40S / RTX 6000 Ada 48GB（832×480、81 framesを基準）
- 24GB: 動く場合はありますが長尺・RIFE・upscale用途には非推奨

長尺4セクションを一度に走らせず、5秒ごとに結果を確認してください。RIFEとupscaleは生成が完成してから有効化する方がVRAMと時間を無駄にしません。

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
