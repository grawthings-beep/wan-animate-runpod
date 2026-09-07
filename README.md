# WAN 2.2 Seamless Loop for RunPod

WAN 2.2 Smooth v6由来のシームレスループに、Enhanced V2 Q8 High/Low（Lightning内蔵）を組み込んだRunPod bundleです。モデルを巨大なDocker imageへ埋め込まず、公式RunPod ComfyUI imageと固定済みcustom nodesを先にpullし、Pod起動後に検証済みモデルを4本並列で取得します。

## v2の構成

4090と5090を同じCUDA imageで動かしません。

| GPU | Container image | CUDA / Torch |
|---|---|---|
| RTX 4090系 | `ghcr.io/grawthings-beep/wan-animate-runpod:loop-ada-cu128-sha-<commit>` | CUDA 12.8 / Torch 2.10 cu128 |
| RTX 5090・Blackwell系 | `ghcr.io/grawthings-beep/wan-animate-runpod:loop-blackwell-cu130-sha-<commit>` | CUDA 13.0 / Torch 2.10 cu130 |

本番では可変tagではなく、GitHub Actionsが発行する40文字commit付きtagを使います。GPUとimageを間違えた場合、またはCUDA 13に必要な580未満のdriver hostへ割り当てられた場合は、大容量モデルを取る前に明示的に停止します。

Docker imageにはモデルを含めません。WAN本体とLoRAをOCI layerへ入れると、モデル1本の変更でも巨大layerのpull・展開・registry cacheが発生し、今回の「Network Volumeなし・毎回新規取得」では不利だからです。Enhanced本体2本（計30.81 GB）はCivitai指定file IDとSHA-256・サイズが一致するrevision固定済みHugging Face配信からHF Xetで取得し、失敗時はaria2へfallbackします。元のSmoothMixはlegacy profileに残し、そちらは複数の同一SHA mirrorへのfailoverも維持します。

## Enhanced V2への移行

- 指定モデルは[High 2584698 / file 2472092](https://civitai.com/models/2053259?modelVersionId=2584698)と[Low 2584707 / file 2472025](https://civitai.com/models/2053259?modelVersionId=2584707)。別版のSVI/Fast Moveとは混同しません。
- 通常I2Vと全8本のloop派生を`UnetLoaderGGUF`へ変更。ファイル名の`smooth_v6`は既存利用との互換性のため残しています。新しいimageでも、ブラウザで開きっぱなしの古いcanvasは自動更新されないため、同梱workflowを開き直してください。
- [作者の推奨](https://civitai.com/models/2053259?modelVersionId=2584698)に合わせ、High **2** + Low **3** steps、Euler/simple、CFG **1**。両samplerの`steps=5`、Highは`0→2`、Lowは`2→5`です。既存loopと同じshift 8を両側で共有します。
- Lightning内蔵なので、外付けLightX2V/Lightningをloopから除外。NAGも外した標準sampler構成です。**CFG 1ではnegative promptは効きません。**
- [ComfyUI-GGUFはLoRA patchに対応](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/nodes.py)。手持ちの追加LoRAは選択肢に残しますが、初期値は全てOFFです。アーキテクチャの互換性と画質の相性は別で、従来の強度が新しいmergeで最適とは限りません。実際の各LoRAの画質はGPU生成での確認が必要です。
- 最初と最後の同一画像conditioning、AIアップスケール、RIFE、後処理モザイク、batch10の逐次queue/ZIP保存は維持します。endpoint指定はループを助けますが、継ぎ目の自然さまでは保証しません。
- 速度比較用の2+2設定は**両方の`steps=4`、High `0→2`、Low `2→4`**。LoRA patchや量子化展開、offloadのコストがあるので、Q8がFP8より速いとは限りません。まず同一画像・固定seedで比較してください。

## Model profile

| profile | assets | download | 内容 |
|---|---:|---:|---|
| `loop-core` | 13 | 41.58 GB | Enhanced V2 Q8、NSFW-22/SmoothXXXAnimationの2組＋wind（OFF）、encoder、VAE、RIFE、モザイク検出器、NMKD-Siax |
| `loop-all` | 29 | 47.36 GB | coreと従来の追加LoRA全組（OFF）。外付けLightX2Vは不要 |
| `loop-quality` | 29 | 47.36 GB | 旧設定互換の`loop-all` alias |

追加LoRAを2組＋windに絞るなら`MODEL_PROFILE=loop-core`、手持ち全組を選びたいなら`loop-all`です。core workflowにはprofile外のLoRA行を置かないため、意図的に省いたモデルの不足警告を防ぎます。

### Wind motion LoRA

[作者版 v1.0](https://civitai.com/models/1865813?modelVersionId=2111773)の`wind.safetensors`を追加しています。Low-noise学習の単体LoRAなので、全8本のloop派生と通常I2V版では**LOW LORA LOADERだけに1行、1.0 / OFF**で配置。試すときはこの行をONにします。1.0は編集可能な初期値であり、作者推奨の最適値ではありません。High側へ同名ファイルを自動追加しません。

HF上のファイル名は`lownoise.safetensors`ですが、作者版と同一SHA-256を確認した固定revisionから取得し、ComfyUIには`models/loras/wind.safetensors`として配置します。約307 MB、追加Secretやcustom nodeは不要。HF Xet優先・aria2 fallback・ダウンロード後SHA-256検証を既存の経路で行います。固有トリガーの登録はありません。Enhanced V2での生成品質とループの継ぎ目は実GPUでの確認が必要です。

loop系はLanczos拡大ではなく、デコードした各フレームを`4x_NMKD-Siax_200k`で4倍AIアップスケールしてから`nearest-exact`で0.5倍へ戻すため、最終サイズは従来どおり実質2倍です。標準のComfyUIノードだけを使い、追加ダウンロードは約67 MBです。

## Bundled workflows

- `wan22_smooth_v6_i2v_auto_mosaic_runpod.json`（開始画像1枚の通常I2V。ループ条件なし、AIアップスケール・RIFE・CPU輪郭モザイク込み）
- `wan22_smooth_v6_seamless_loop_core_runpod.json`
- `wan22_smooth_v6_seamless_loop_core_auto_mosaic_runpod.json`
- `wan22_smooth_v6_seamless_loop_batch10_core_runpod.json`
- `wan22_smooth_v6_seamless_loop_batch10_core_auto_mosaic_runpod.json`
- 上記4本の全LoRA版（ファイル名に`core`なし）
- AIOとNative Enhanced Lightningは互換用。通常のpushではlegacy full imageを作らず、手動Actionだけで作ります。

batch10は10本を同時にGPUへ載せません。専用の一括投入欄へ10枚入りフォルダ（またはZIP）と`prompts.txt`をdropすると、自然なファイル名順で10slotを自動設定します。専用ボタンを1回押すと独立した10 jobを順番にqueueし、10本目の完了後だけZIPを自動downloadします。

`prompts.txt`は1件のpromptを何行でも書けます。次の画像用promptとの間に空行を1行以上入れ、合計10ブロックにしてください。画像は`01.png`〜`10.png`のように命名します。旧形式の1行×10件、JSON文字列10件の配列、`---`区切りにも対応します。ZIP内に`prompts.txt`を含めればZIP 1個のdropだけで設定完了です。

auto-mosaic版は完成frameにCPUのYOLO11 segmentationを適用し、RIFE後・MP4 encode前で輪郭に沿ったモザイクを入れます。既定対象は`pussy,penis,testicles`で、`anus`には適用しません。WANとVRAMを奪い合いません。

通常I2Vモザイク版は開始画像を1枚だけ指定します。First/Last Frameや開始地点へ戻すconditioningを持たないため、登場・退場・一方向の動作など、ループに向かない時系列の動画をそのまま生成できます。Deepthroat/Face Fuck v3の`Wan22_ThroatV3_High` / `Wan22_ThroatV3_Low`もLoRA欄に1.0・OFFで収録済みです。

## 起動の流れ

```text
8188 status page
  -> GPU / image / driver / Torch実演算検査
  -> workflow配置
  -> disk容量検査
  -> 大きいモデルから4本並列取得（HF Xet + aria2 + 同一SHA mirror failover）
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

- 11 workflowの再生成差分と58 asset manifestの整合検査
- Python unit tests、JavaScript構文、shell構文
- Ada/cu128とBlackwell/cu130を2 job並列build
- 各image内で本番`start.sh`を`--quick-test-for-ci`実行し、custom node import、CLI、writable user/workflow pathを検査
- 両image内で小さなQ8 tensorへ合成LoRAを適用し、clone・forwardの実計算・解除までCPU検査。14B本体のGPU生成品質テストではありません
- 高コストだったGitHub Actions cache exportを廃止し、profile別GHCR registry cacheを利用

## Local validation

```bash
python scripts/prepare_workflows.py --check
python scripts/validate_assets.py
python -m unittest discover -s tests -v
bash -n scripts/common.sh scripts/install_custom_nodes.sh scripts/start.sh scripts/container_smoke.sh
```
