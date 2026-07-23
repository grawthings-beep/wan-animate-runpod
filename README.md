# WAN 2.2 Smooth Workflow v6.0 on RunPod

RunPod上で **SmoothMix WAN 2.2 I2V/T2V、First-to-Last Frame、MMAudio、RIFE補間**を動かすためのComfyUIイメージです。元のLTX構成は廃止し、提供された `WAN 2.2 Smooth Workflow v6.0 AIO` に合わせて刷新しています。

## この構成でできること

- I2V: SmoothMix I2V v2 High/Low + 作者推奨LightX2V rank128
- T2V: 最新SmoothMix T2V v4 High/Low（LightX2V内蔵）
- Seamless Loop: 同一画像を先頭・末尾フレームにした専用プリセット
- Audio: MMAudio 44k v2 + BigVGANによる同期効果音
- Finish: RIFE補間、2倍アップスケール、H.264 MP4出力
- Full profile: ワークフローから参照・案内されるモデル／LoRAを全自動取得

モデルはコンテナに焼かず、初回起動時に永続ボリュームへダウンロードします。大きいファイルから4本を並列取得し、Hugging FaceはRust製 `hf_xet` の適応型並列転送、CivitAIはaria2の分割転送を使います。各ファイルはサイズとSHA256で検証され、途中でPodが止まってもHubキャッシュまたはaria2の `.part` から再開します。カスタムノードは14個すべてコミット固定です。

## コンテナイメージ

`main` へのpushでGitHub Actionsが次を発行します。

```text
ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6
ghcr.io/grawthings-beep/wan-animate-runpod:cuda12.8
ghcr.io/grawthings-beep/wan-animate-runpod:latest
```

RunPodから認証なしでpullする場合、GitHubのPackages画面でこのGHCRパッケージを **Public** にしてください。

## RunPod推奨設定

| 項目 | 推奨値 |
|---|---|
| Template type | Pod |
| Container image | `ghcr.io/grawthings-beep/wan-animate-runpod:wan22-smooth-v6` |
| Container disk | 50 GB |
| Volume disk | 250 GB（`full`用。生成物を多く残すなら300 GB） |
| Volume mount path | `/workspace` |
| HTTP port | `8188` |
| GPU | A100 80GB / H100 80GB（最高品質）、L40S / RTX 6000 Ada 48GB（実用下限） |

48GBでは480×832前後のベース解像度が現実的です。80GBなら600×900付近、長めのフレーム列、MMAudioまで余裕を持って扱えます。24GBでもCPUオフロードで動く場合はありますが、高品質動画用途では非常に遅くなります。

環境変数:

```text
PORT=8188
LISTEN=0.0.0.0
DOWNLOAD_MODELS=1
MODEL_PROFILE=full
CIVITAI_API_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
RUN_DEP_CHECK=1
DOWNLOAD_WORKERS=4
ARIA2_CONNECTIONS=8
ARIA2_SPLITS=8
HF_XET_HIGH_PERFORMANCE=1
COMFYUI_ARGS=--reserve-vram 3
```

`full` はCivitAI上の最新T2V v4とSmoothMix LoRAも取得するため、`CIVITAI_API_TOKEN` が必須です。この構成ではRunPod Secret名を `CIVITAI_TOKEN`、`HF_TOKEN` とし、環境変数欄の鍵アイコンからそれぞれ割り当てます。Hugging Faceのファイル自体は公開ですが、`HF_TOKEN` を渡すと匿名リクエストのレート制限を避けられます。

`MODEL_MANIFEST_URL` は設定不要です。イメージ内に検証済みmanifestを同梱しています。

## モデルプロファイル

| `MODEL_PROFILE` | 内容 | 用途 |
|---|---|---|
| `full` | 全モデル、全案内LoRA、MMAudio、RIFE、代替GGUF | 要望どおり全部入れる既定値 |
| `i2v-quality` | I2V High/Low、必須LightX2V、共有モデル、RIFE | I2Vだけを軽く始める |
| `loop-quality` | ループ枝が参照する26資産（I2V、追加LoRA、MMAudio、RIFE、代替GGUF）。T2V専用High/Lowのみ除外 | 不足モデルなしでループを使う（約65GB） |
| `t2v-quality` | T2V v4 High/Low、共有モデル、RIFE | T2Vだけを使う |

`full` は約98GB、`loop-quality` は約65GBです。速度はRunPodホスト、永続ボリューム、Hugging Face/CivitAI側の混雑に左右されますが、4ファイルを並行し、各ファイル内も分割・適応並列化して回線の遊休時間を減らします。モデル、ComfyUIユーザーデータ、入力、出力はすべて `/workspace` 以下に置かれ、Pod交換後も残ります。

## ワークフロー

初回起動後、ComfyUIの **Workflows → Open** から選べます。

- `wan22_smooth_v6_aio_runpod`: I2V / T2V / First2LastFrame / Audio2VideoのAIO
- `wan22_smooth_v6_seamless_loop_runpod`: First2LastFrameを選択済みのループ専用版

I2V用Power LoRA Loaderには、作者推奨の次の設定をあらかじめ入れています。

```text
High noise: lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16 = 3.0
Low noise:  lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16 = 1.5
```

このLoRAを外すと、I2V v2はぼけ・ノイズ化しやすくなります。T2V v4にはLightX2Vが内蔵されているため、追加の加速LoRAは既定OFFです。

## 高品質I2Vの始め方

1. AIOを開き、上部の `Worflows` スイッチで `IMAGE2VIDEO` を選択します。
2. 入力画像とプロンプトを指定します。
3. 最初は `480×832` または `512×896`、81 frames、Shift 8、CFG 1、High 4 steps + Low 4 stepsから試します。
4. 動きが弱ければShiftを6へ、形状維持を優先するなら10へ寄せます。
5. 短い生成が安定してから解像度、フレーム数、RIFE、2倍アップスケールを上げます。

900×600のT2V v4ショーケース設定はSteps 8、Euler、Simpleです。まず低いベース解像度で構図・動きを確定し、最後に補間とアップスケールを行う方が失敗コストを抑えられます。

## シームレスループ

専用ワークフローを開き、`FIRST FRAME` と `LAST FRAME` の両方に **まったく同じ画像**を指定します。プロンプトは「呼吸、揺れ、回転、波、脈動」など元の状態へ戻れる周期運動にし、カット、登場・退場、一方向の移動、不可逆な変形は避けます。

最初は81 framesの短いループを作り、継ぎ目を確認してからRIFEとアップスケールを有効にしてください。専用版は音の継ぎ目を作らないためMMAudioを既定OFFにしています。MP4自体には無限再生指定がないため、再生側でloopを有効にします。

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

## トラブルシューティング

- 起動直後にCivitAI tokenエラー: RunPod Secret名と `CIVITAI_API_TOKEN` の割り当てを確認します。
- 黒画面・崩れた映像: 古いComfyUIで起きやすいため、このイメージの固定バージョンを使い、別の古いPodからcustom nodeを持ち込まないでください。
- I2Vがぼける: rank128 LightX2VがHigh=3.0、Low=1.5でONか確認します。
- モデルが表示されない: ログ末尾の `[check_env] ... missing=0` を確認します。新しいイメージは既存ボリューム上の古い標準manifestを起動ごとに更新します。
- 初回起動が長い: `full` は約98GBです。ログの `TRANSFER ENGINE`、`HF_XET:`、aria2進捗を確認してください。帯域制限や429が出る環境では `DOWNLOAD_WORKERS=2` に下げると安定します。
- OOM: ベース解像度、frames、RIFE batchを下げるか、80GB GPUへ上げます。
