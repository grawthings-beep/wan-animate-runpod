# Workflows

- `wan22_smooth_v6_aio_runpod.json`: I2V / T2V / First2LastFrame / Audio2Videoの全機能版。
- `wan22_smooth_v6_seamless_loop_runpod.json`: First2LastFrameを有効化し、音声を無効化したシームレスループ版。
- `source/WAN 2.2 Smooth Workflow v6.0.json`: ユーザー提供の未加工ソース。

生成版は `scripts/prepare_workflows.py` で作ります。モデルの実ファイル名、I2V LightX2V High=3.0 / Low=1.5、RunPod用メタデータを決定的に適用しています。

```bash
python scripts/prepare_workflows.py
python scripts/prepare_workflows.py --check
```

Pod起動時にトップレベルの生成版JSONだけが `/workspace/comfyui/user/default/workflows/` へコピーされます。既存のユーザー編集は上書きしません。
