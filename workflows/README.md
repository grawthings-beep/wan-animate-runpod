# Workflows

- `wan22_native_enhanced_lightning_longvideo_runpod.json`: 5〜20秒の連続I2V、RIFE、upscaleを含む今回のworkflow
- `wan22_smooth_v6_aio_runpod.json`: Smooth v6のI2V / T2V / First-to-Last Frame / Audio2Video
- `wan22_smooth_v6_seamless_loop_runpod.json`: Smooth v6のシームレスループpreset
- `source/*.json`: ユーザー提供の未加工source

生成物は`scripts/prepare_workflows.py`で決定的に作ります。配布モデル名への正規化、古い絶対パスとvideo preview metadataの削除、任意LoRA表示、RunPod profile metadataの追加を行います。

```bash
python scripts/prepare_workflows.py
python scripts/prepare_workflows.py --check
```

起動時、生成済みの`*_runpod.json`だけが`/workspace/comfyui/user/default/workflows/`へ入ります。既存のユーザー編集版は上書きせず、bundle内容が変わった場合はhash付きの新ファイルを追加します。
