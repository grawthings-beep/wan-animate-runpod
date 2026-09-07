# 5秒ループの速度・モデル再検討（2026-09-08）

## 確認できたこと / まだ分からないこと

ユーザー申告は720×960、AIアップスケールありで272秒。GPU型番、当該生成ログ、初回/2回目の別がまだないため、異常値とも、チェックポイントが主因とも断定しない。旧標準528×704に対して720×960は画素数が約1.86倍。最終2倍出力なら1440×1920であり、「最終720×960」の測定とは別物。

旧コードの確実な問題は、UpscaleModelLoaderのwidgets_valuesが配列でなく文字列だったこと。LiteGraphが先頭文字をwidget 0として復元し、モデル名が「4」になっていた。文字列配列へ修正し回帰テストを追加。

旧経路は81枚を4倍RRDBで拡大し、全4倍フレームを保持してから2倍へ縮小、RIFE x2（ensemble ON）、CPUで全補間フレームのモザイク検出。720×960入力なら4倍中間テンソルだけで約10.75 GB（81枚、RGB、float32）。これ以外にも入力・縮小後・補間後・モザイク出力がある。CPU/RAM/offload時間をGPUの推論速度と混同できない。

## CivArchiveの作者記載を比較

| 候補 | 作者記載 | 今回の判断 |
|---|---|---|
| [Enhanced V2 Q8](https://civarchive.com/models/2053259?modelVersionId=2584698) | Lightning内蔵、Euler/simple・CFG1・2+2、2+3は画質寄り | 既に低ステップモデル。今回2+3から2+2へ変更。272秒全体が20%短くなるという意味ではない |
| [DaSiWa Lightspeed v11 High](https://civarchive.com/models/1981116?modelVersionId=2953474) / [作者workflow](https://civarchive.com/models/1823089?modelVersionId=2712329) | High/Lowペア、4-step、外付け速度LoRA不要。プロンプト追従・動き改善を作者が標榜 | 次の比較候補。作者の宣伝を実測結果として扱わない。今回の14B checkpointは未変更 |
| [SmoothMix I2V v2](https://civarchive.com/models/1995784?modelVersionId=2513186) / [Smooth v6](https://civarchive.com/models/1847730?modelVersionId=2110988) | I2V v2にはLightX2V未統合。作者サンプルはHigh3.0/Low1.5を使用。v6はFirst/Lastによる短いループ例を掲載 | 「LoRAなし・高速」を単純に満たす置換先ではない。T2Vの内蔵LoRA説明をI2Vへ流用しない |

SVIは連続セグメントの長尺生成用。今回の短い閉ループを速くする目的だけで導入しない。作者自身も動き・プロンプト追従の弱点を挙げている。

[TurboDiffusion本家](https://github.com/thu-ml/TurboDiffusion)と[ComfyUI実装](https://github.com/anveshane/Comfyui_turbodiffusion)も確認した。専用蒸留checkpointとrCM/SLAを使う別経路で、READMEのI2V sampler入力に終了画像はない。現在のFirst/Last条件・既存LoRAとの互換を確認せず「100倍高速」として入れ替えることはしない。通常の40-step等との比較倍率を、既に4-stepの本構成にそのまま当てはめられない。

## 今回実装した変更

- 配布は`wan22_loop_single_runpod.json`と`wan22_loop_batch10_runpod.json`の2本。両方モザイクあり。
- 生成解像度720×960、High2+Low2、既存の同一画像First/Last条件を維持。
- [Helaman作2x NomosUni SPAN multijpg](https://openmodeldb.info/models/2x-NomosUni-span-multijpg)（CC-BY-4.0）を既定にし、取得revision/SHA-256を固定。4.5 MBのsafetensors配信。旧NMKDも選べる。
- 標準ComfyUIのタイル処理を1フレームずつ使用。異なる倍率のモデルでも直ちに指定最終寸法に合わせ、4倍全動画の中間保持を廃止。
- 順番をdecode→RIFE→SPAN→モザイク→MP4に変更。RIFEは拡大前の720×960で処理するため、入力画素数は旧順序の1/4。その代わりSPANは補間後の約161枚を処理する。出力解像度・フレーム数は維持し、重い補間の解像度を抑える設計。総実行時間の短縮率は未測定。
- RIFE x2は維持、ensemble OFF・batch_size=1。ensemble OFFと補間/拡大の順序変更は速度/品質のトレードオフであり同一画質を保証しない。
- モザイクは全フレーム検出を維持。生成後の空きGPUを利用し、不足/OOM時はCPU。検出対象・輪郭範囲を勝手に緩めない。
- `[wan-post] upscale ... seconds=` / `[wan-post] mosaic ... detect_seconds=... apply_seconds=...`を追加。

VAE2xも調べたが、[作者が画像学習・実写主体でアニメ/線画が苦手と明記](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x)しているため、今回の既定には選ばなかった。

## LoRAなしで衣服を風になびかせる比較

風表現を特定LoRAの必須機能として扱わない。ただし、入力画像の構図・衣服、プロンプト、seed、同じ画像で両端を拘束する影響があり、どのcheckpointでも成功するとは断定できない。今回のwind LoRAはOFFのまま残す。

非性的な比較プロンプト例（モデル作者のトリガーではない）:

```text
A steady crosswind continuously ripples the skirt fabric and loose hair.
The hem sways outward and settles back in repeating flowing waves.
The person maintains the same pose while the fabric keeps moving.
Fixed camera, continuous natural motion, consistent lighting.
```

モデル比較は同じ入力、720×960、81 frames、同一seed群で行う。まず追加LoRAは全OFF。ループあり/通常I2Vも比較して、終端拘束による動きの弱さとモデル自体の問題を分離する。1回の成功例だけで優劣を決めない。

時間は①初回ロード込み、②同条件2回目、③sampler、④upscale、⑤RIFE、⑥mosaic、⑦encodeを区別する。初回とwarmの混在、32GB desktop 5090と24GB laptop 5090の混在、最終寸法と生成寸法の混在を避ける。

この変更で実施できるCIは構成・小規模CPU推論・模擬OOMの検査。実GPUでの272秒からの短縮幅、LoRAの相性、衣服の動きとループの自然さは未検証。
