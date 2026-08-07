# WAN 2.2 seamless loop — 10本逐次生成

対象ワークフロー:

`wan22_smooth_v6_seamless_loop_batch10_runpod.json`

完成動画へ自動モザイクを入れる場合は、同じ操作で`wan22_smooth_v6_seamless_loop_batch10_auto_mosaic_runpod.json`を使います。各ジョブのRIFE後・MP4保存前にCPU処理し、最後のZIPにはモザイク済み10本だけが入ります。

## 使い方

1. `01`〜`10` の各スロットで画像を1枚選び、その画像に対応するpositive promptを入力します。
2. 紫色の `QUEUE 10 LOOPS — ONE JOB AT A TIME` ノードにある `QUEUE 10 LOOPS (SEQUENTIAL)` ボタンを1回だけ押します。
3. ComfyUIのQueueには独立した10ジョブが入り、1本ずつ順番に生成されます。通常の画面上部にあるQueueボタンは、この操作には使いません。
4. 10本目まで成功すると、10本のMP4とprompt一覧を記録した `manifest.json` を含むZIPが自動ダウンロードされます。

各スロットの画像は、そのジョブ内でfirst frameとlast frameの両方へ同じものが渡されます。そのため、生成動画の末尾に別の開始画像を後付けする構成ではなく、元のFirst-to-Last Frameループ生成を保ったまま継ぎ目を作ります。

## OOMを避ける仕組み

画像10枚をIMAGE batchやlistとして一度に推論へ渡していません。ボタンはComfyUIの通常ジョブを10件作成し、各ジョブは選択中の1画像だけをロードします。`WanFirstLastFrameToVideo` の `batch_size` も `1` のままです。

途中のジョブが失敗した場合、ComfyUIの残りのQueueを直して再実行してください。10番だけ先に完了しても、1〜9番の成果物が揃っていなければZIPは作られません。再実行時は新しいbatch IDになるため、10件を最初からキュー登録するのが安全です。

## 保存先

```text
/workspace/comfyui/output/Video/loop-batches/<batch-id>/
├── slot-01_....mp4
├── ...
├── slot-10_....mp4
├── manifest.json
└── <batch-id>.zip
```

ZIPはMP4を再圧縮せずまとめるため、10本目の完了後に余計な長時間エンコードは発生しません。ブラウザ設定で自動ダウンロードが抑止された場合は、緑色の最終ノードにある `DOWNLOAD LAST ZIP` を押してください。上記フォルダのZIPはRunPodのファイル操作からも取得できます。
