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

## 2本に統一したループ構成

- 指定モデルは[High 2584698 / file 2472092](https://civitai.com/models/2053259?modelVersionId=2584698)と[Low 2584707 / file 2472025](https://civitai.com/models/2053259?modelVersionId=2584707)。別版のSVI/Fast Moveとは混同しません。
- 同梱は単体ループと10本逐次生成だけ。両方とも自動モザイク・AIアップスケール付きです。ブラウザで開きっぱなしの古いcanvasは更新されないため、新しい同梱workflowを開き直してください。
- [作者の推奨](https://civarchive.com/models/2053259?modelVersionId=2584698)の速度寄り設定、High **2** + Low **2** steps、Euler/simple、CFG **1**。両samplerの`steps=4`、Highは`0→2`、Lowは`2→4`です。shift 8を両側で共有します。
- Lightning内蔵なので、外付けLightX2V/Lightningをloopから除外。NAGも外した標準sampler構成です。**CFG 1ではnegative promptは効きません。**
- [ComfyUI-GGUFはLoRA patchに対応](https://github.com/city96/ComfyUI-GGUF/blob/6ea2651e7df66d7585f6ffee804b20e92fb38b8a/nodes.py)。手持ちの追加LoRAは選択肢に残しますが、初期値は全てOFFです。アーキテクチャの互換性と画質の相性は別で、従来の強度が新しいmergeで最適とは限りません。実際の各LoRAの画質はGPU生成での確認が必要です。
- 最初と最後の同一画像conditioning、AIアップスケール、RIFE、後処理モザイク、batch10の逐次queue/ZIP保存は維持します。endpoint指定はループを助けますが、継ぎ目の自然さまでは保証しません。
- 画質比較用の2+3設定は**両方の`steps=5`、High `0→2`、Low `2→5`**。Q8がFP8より速いとは限りません。新しい標準サイズは生成720×960、最終1440×1920。解像度を隠れて下げる高速化はしていません。

## Model profile

| profile | assets | download | 内容 |
|---|---:|---:|---|
| `loop-core` | 14 | 41.58 GB | Enhanced V2 Q8、NSFW-22/SmoothXXXAnimationの2組＋wind（OFF）、encoder、VAE、RIFE、モザイク、SPAN/NMKD |
| `loop-all` | 30 | 47.36 GB | coreと従来の追加LoRA全組（OFF）。外付けLightX2Vは不要 |
| `loop-quality` | 30 | 47.36 GB | 旧設定互換の`loop-all` alias |

追加LoRAを2組＋windに絞るなら`MODEL_PROFILE=loop-core`、手持ち全組を選びたいなら`loop-all`です。どちらでもworkflow名は同じ2本です。起動時にprofile外のLoRA行だけを除くため、未取得LoRAの不足警告を防ぎます。

### Wind motion LoRA

[作者版 v1.0](https://civitai.com/models/1865813?modelVersionId=2111773)の`wind.safetensors`は**LOW LORA LOADERに1.0 / OFF**で残しています。作者推奨の最適値やEnhancedとの良好な相性を確認した値ではありません。風表現に必須ではなく、まずOFFで比較します。

HF上のファイル名は`lownoise.safetensors`ですが、作者版と同一SHA-256を確認した固定revisionから取得し、ComfyUIには`models/loras/wind.safetensors`として配置します。約307 MB、追加Secretやcustom nodeは不要。HF Xet優先・aria2 fallback・ダウンロード後SHA-256検証を既存の経路で行います。固有トリガーの登録はありません。Enhanced V2での生成品質とループの継ぎ目は実GPUでの確認が必要です。

標準アップスケーラーは[Helaman作2x NomosUni SPAN multijpg](https://openmodeldb.info/models/2x-NomosUni-span-multijpg)（CC-BY-4.0、約4.5 MBのsafetensors配信）。Hugging FaceのrevisionとSHA-256を固定しています。モデル名widgetは正しく配列保存し、先頭の「4」だけになる不具合を修正しました。`WanLoopModelUpscale`が標準ComfyUIのタイル処理を1フレームずつ呼び、4倍の全動画を中間保持しません。旧NMKDも選択可能で、その場合は1フレームごとに最終2倍へ戻します。

後処理は **decode → RIFE → AI upscale → mosaic → MP4**。RIFEを拡大前の720×960で処理し、1440×1920で補間する旧順序を変更しました。RIFE x2は維持し、ensembleの追加往復計算はOFF、GPU batchは1です。拡大する枚数は約161枚に増えますが、補間に渡す画素数は1/4です。画質と総実行時間は実GPUで比較が必要です。

## Bundled workflows

- `wan22_loop_single_runpod.json`：単体ループ＋AIアップスケール＋自動モザイク
- `wan22_loop_batch10_runpod.json`：10本逐次生成＋同じ後処理＋ZIP

旧11本とhash付きbundleコピーは配布終了。新イメージ起動時に、既知の旧名と更新前の編集済み同梱名を`/workspace/comfyui/workflow-backups/`へ退避します。任意の名前で保存したユーザーworkflowは移動しません。旧repo版はGit履歴から復元できます。

batch10は10本を同時にGPUへ載せません。専用の一括投入欄へ10枚入りフォルダ（またはZIP）と`prompts.txt`をdropすると、自然なファイル名順で10slotを自動設定します。専用ボタンを1回押すと独立した10 jobを順番にqueueし、10本目の完了後だけZIPを自動downloadします。

`prompts.txt`は1件のpromptを何行でも書けます。次の画像用promptとの間に空行を1行以上入れ、合計10ブロックにしてください。画像は`01.png`〜`10.png`のように命名します。旧形式の1行×10件、JSON文字列10件の配列、`---`区切りにも対応します。ZIP内に`prompts.txt`を含めればZIP 1個のdropだけで設定完了です。

モザイクはRIFE後・MP4 encode前の全フレームへ適用します。既定対象とJUST輪郭の範囲は変更していません。`device=auto`ではWAN完了後にComfyUI管理モデルをoffloadし、空きVRAMが2 GiB以上ならGPUを使います。GPU OOM時はそのフレームからCPUで再試行。`device=cpu`も選べます。間引き検出はしません。ログの`[wan-post]`にアップスケールとモザイクの処理秒数を出します。

272秒の申告と候補モデルの再調査は[速度・モデル再検討メモ](docs/loop-performance-review.md)へ整理しています。実GPUでの速度改善率・衣服の動き・継ぎ目品質は未測定です。

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

- 2 workflowの再生成差分と59 asset manifestの整合検査
- Python unit tests、JavaScript構文、shell構文
- Ada/cu128とBlackwell/cu130を2 job並列build
- 各image内で本番`start.sh`を`--quick-test-for-ci`実行し、custom node import、CLI、writable user/workflow pathを検査
- 両image内で小さなQ8 tensorへ合成LoRAを適用し、clone・forwardの実計算・解除までCPU検査。14B本体のGPU生成品質テストではありません
- 両image内で実際のSPAN weightsを取得・ハッシュ検証・小画像の2倍推論を検査。4倍モデル選択時の出力寸法維持と、モザイクGPU OOM→CPU再試行も検査（OOMは模擬）
- 高コストだったGitHub Actions cache exportを廃止し、profile別GHCR registry cacheを利用

## Local validation

```bash
python scripts/prepare_workflows.py --check
python scripts/validate_assets.py
python -m unittest discover -s tests -v
bash -n scripts/common.sh scripts/install_custom_nodes.sh scripts/start.sh scripts/container_smoke.sh
```
