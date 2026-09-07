# Workflows

配布するワークフローは次の2本だけです。両方とも同じ生成・後処理設定です。

- `wan22_loop_single_runpod.json`: 1本のFirst/Last画像ループ。
- `wan22_loop_batch10_runpod.json`: 10組の画像・promptを独立ジョブとして順番に処理し、完了後に動画入りZIPを自動ダウンロード。

共通: Enhanced V2 Q8、4 steps（High 2 + Low 2）、720×960、約5秒、軽量2x SPANモデルによる1440×1920へのアップスケール、RIFE x2、完成フレームへのJUST輪郭モザイク。追加LoRAは初期値OFF。モザイクは全フレームを検出し、`auto`では空きGPUメモリを確認して使用、GPU OOM時はそのフレームからCPUで再試行します。検出漏れがないことを保証する仕組みではないため、共有前に出力を確認してください。

処理順はdecode → RIFE（拡大前）→ 2x SPAN → モザイク → MP4です。補間を高解像度で処理していた旧版とは順序を変えています。

`MODEL_PROFILE=loop-core`でもファイル名は同じです。起動時に未ダウンロードの任意LoRA行だけを取り除き、別のcore版は追加しません。`loop-all` / `loop-quality`では全LoRAを選択できます。

生成物は`scripts/prepare_workflows.py`で決定的に作ります。`source/*.json`はユーザー提供の原本で、Podの一覧にはインストールしません。

```bash
python scripts/prepare_workflows.py
python scripts/prepare_workflows.py --check
```

起動時は`scripts/install_workflows.py`が上記2本を`/workspace/comfyui/user/default/workflows/`へ配置します。以前の配布名・hash付きコピーと、内容が異なる同名の編集版は、一覧の外の`/workspace/comfyui/workflow-backups/`に内容hash付きで退避してから更新します。独自の名前で保存したワークフローは触りません。過去の配布JSONもGit履歴から復元できます。

GPUごとの実測速度は未確認です。変更理由とチェックポイント調査は[性能レビュー](../docs/loop-performance-review.md)、10本版の操作は[使い方](../BATCH10_WORKFLOW.md)を参照してください。
