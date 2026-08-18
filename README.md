# WAN 2.2 Seamless Loop for RunPod

WAN 2.2 Smooth v6のシームレスループを、RunPodで毎回クリーンに立ち上げるためのbundleです。モデルを巨大なDocker imageへ埋め込まず、公式RunPod ComfyUI imageと固定済みcustom nodesを先にpullし、Pod起動後に検証済みモデルを4本並列で取得します。

## v2の構成

4090と5090を同じCUDA imageで動かしません。

| GPU | Container image | CUDA / Torch |
|---|---|---|
| RTX 4090系 | `ghcr.io/grawthings-beep/wan-animate-runpod:loop-ada-cu128-sha-<commit>` | CUDA 12.8 / Torch 2.10 cu128 |
| RTX 5090・Blackwell系 | `ghcr.io/grawthings-beep/wan-animate-runpod:loop-blackwell-cu130-sha-<commit>` | CUDA 13.0 / Torch 2.10 cu130 |

本番では可変tagではなく、GitHub Actionsが発行する40文字commit付きtagを使います。GPUとimageを間違えた場合、またはCUDA 13に必要な580未満のdriver hostへ割り当てられた場合は、46 GBを取る前に明示的に停止します。

Docker imageにはモデルを含めません。WAN本体とLoRAをOCI layerへ入れると、モデル1本の変更でも巨大layerのpull・展開・registry cacheが発生し、今回の「Network Volumeなし・毎回新規取得」では不利だからです。

## Model profile

| profile | assets | download | 内容 |
|---|---:|---:|---|
| `loop-core` | 12 | 40.96 GB | 実際にONのLightX2V、NSFW-22、SmoothXXXAnimation High/Low、WAN本体、encoder、VAE、RIFE、モザイク検出器 |
| `loop-all` | 28 | 46.74 GB | coreにCumshot、JOI、Deepthroat/Face Fuck v3、iroiro 5組のOFF LoRAを追加 |
| `loop-quality` | 28 | 46.74 GB | 旧設定互換の`loop-all` alias |

最短起動なら`MODEL_PROFILE=loop-core`、追加済みLoRAを画面から選びたいなら`loop-all`です。core workflowにはOFF LoRAの行自体がないため、不足モデル警告も出ません。

## Bundled workflows

- `wan22_smooth_v6_seamless_loop_core_runpod.json`
- `wan22_smooth_v6_seamless_loop_core_auto_mosaic_runpod.json`
- `wan22_smooth_v6_seamless_loop_batch10_core_runpod.json`
- `wan22_smooth_v6_seamless_loop_batch10_core_auto_mosaic_runpod.json`
- 上記4本の全LoRA版（ファイル名に`core`なし）
- AIOとNative Enhanced Lightningは互換用。通常のpushではlegacy full imageを作らず、手動Actionだけで作ります。

batch10は10本を同時にGPUへ載せません。専用の一括投入欄へ10枚入りフォルダ（またはZIP）と`prompts.txt`をdropすると、自然なファイル名順で10slotを自動設定します。専用ボタンを1回押すと独立した10 jobを順番にqueueし、10本目の完了後だけZIPを自動downloadします。

`prompts.txt`は1件のpromptを何行でも書けます。次の画像用promptとの間に空行を1行以上入れ、合計10ブロックにしてください。画像は`01.png`〜`10.png`のように命名します。旧形式の1行×10件、JSON文字列10件の配列、`---`区切りにも対応します。ZIP内に`prompts.txt`を含めればZIP 1個のdropだけで設定完了です。

auto-mosaic版は完成frameにCPUのYOLO11 segmentationを適用し、RIFE後・MP4 encode前で輪郭に沿ったモザイクを入れます。既定対象は`pussy,penis,testicles`で、`anus`には適用しません。WANとVRAMを奪い合いません。

## 起動の流れ

```text
8188 status page
  -> GPU / image / driver / Torch実演算検査
  -> workflow配置
  -> disk容量検査
  -> 大きいモデルから4本並列取得（HF Xet + aria2）
  -> SHA-256・custom node・CUDA検査
  -> 同じ8188をComfyUIへhandoff
```

起動直後からRunPodのConnectボタンで8188を開けます。まだComfyUIが起動していなくても、現在のphase、asset数、検証済みGB、失敗理由が表示されます。失敗ページは既定で15分保持します。

同一PodをStop/Startした場合、RunPodのVolume Disk上の`/workspace`が残っていれば検証だけで再利用します。PodをTerminateして新しく作れば再downloadです。Network Volumeは不要です。

## RunPod設定

手順と全環境変数は[RUNPOD_STEPS.md](RUNPOD_STEPS.md)と[runpod-template.env.example](runpod-template.env.example)にあります。

最低限:

1. GPUに合うimmutable image tagを指定。
2. Volume Diskは`loop-core`なら80 GB以上、`loop-all`なら100 GB推奨。
3. HTTP Portに`ComfyUI / 8188`を追加。
4. 環境変数へexampleを貼り、`HF_TOKEN`と`CIVITAI_API_TOKEN`はRunPod Secretsから割り当て。

## CIの保証

mainへのpushごとに以下を実行します。

- 10 workflowの再生成差分と55 asset manifestの整合検査
- Python unit tests、JavaScript構文、shell構文
- Ada/cu128とBlackwell/cu130を2 job並列build
- 各image内で本番`start.sh`を`--quick-test-for-ci`実行し、custom node import、CLI、writable user/workflow pathを検査
- 高コストだったGitHub Actions cache exportを廃止し、profile別GHCR registry cacheを利用

## Local validation

```bash
python scripts/prepare_workflows.py --check
python scripts/validate_assets.py
python -m unittest discover -s tests -v
bash -n scripts/common.sh scripts/install_custom_nodes.sh scripts/start.sh scripts/container_smoke.sh
```
