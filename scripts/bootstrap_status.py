#!/usr/bin/env python3
"""Small dependency-free status page used while WAN assets are provisioned."""

import argparse
import html
import json
import os
import pathlib
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_STATUS = {
    "state": "initializing",
    "phase": "container",
    "message": "WAN loop runtime is initializing.",
}


def read_status(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(DEFAULT_STATUS)
    except (OSError, ValueError, TypeError):
        return dict(DEFAULT_STATUS)


def write_status(path, updates):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_status(path)
    data.update({key: value for key, value in updates.items() if value is not None})
    data["updated_at"] = int(time.time())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return data


def render_page(data):
    state = html.escape(str(data.get("state", "initializing")))
    phase = html.escape(str(data.get("phase", "container")))
    message = html.escape(str(data.get("message", "Initializing")))
    detail = html.escape(str(data.get("detail", "")))
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WAN Loop / {state}</title>
<style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}
body{{margin:0;background:#0d1117;color:#e6edf3;display:grid;min-height:100vh;place-items:center}}
main{{width:min(720px,88vw);background:#161b22;border:1px solid #30363d;border-radius:18px;padding:30px}}
h1{{margin:0 0 18px;font-size:26px}} .phase{{color:#8b949e;text-transform:uppercase;letter-spacing:.12em}}
.bar{{height:10px;background:#30363d;border-radius:99px;overflow:hidden;margin:24px 0}}
.bar span{{display:block;width:45%;height:100%;background:#2f81f7;border-radius:99px;animation:move 1.4s ease-in-out infinite alternate}}
pre{{white-space:pre-wrap;color:#b1bac4}} @keyframes move{{to{{transform:translateX(122%)}}}}
</style></head><body><main>
<div class="phase" id="phase">{phase}</div><h1 id="message">{message}</h1>
<div class="bar"><span></span></div><pre id="detail">{detail}</pre>
<p>このページは自動更新されます。ComfyUIの準備が終わるまで閉じなくて大丈夫です。</p>
</main><script>
async function poll(){{
  try{{
    const r=await fetch('/status.json?ts='+Date.now(),{{cache:'no-store'}});
    if(!r.ok) throw new Error('handoff');
    const s=await r.json();
    document.getElementById('phase').textContent=s.phase||'container';
    document.getElementById('message').textContent=s.message||'Initializing';
    let d=s.detail||'';
    if(s.assets_total) d+='\nAssets: '+(s.assets_completed||0)+' / '+s.assets_total;
    if(s.bytes_total) d+='\nVerified: '+((s.bytes_completed||0)/1e9).toFixed(2)+' / '+(s.bytes_total/1e9).toFixed(2)+' GB';
    document.getElementById('detail').textContent=d.trim();
    if(s.state==='handoff') setTimeout(()=>location.reload(),900);
  }}catch(e){{ setTimeout(()=>location.reload(),1200); return; }}
  setTimeout(poll,1000);
}} poll();
</script></body></html>"""


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(path, host, port):
    path = pathlib.Path(path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            route = self.path.split("?", 1)[0]
            data = read_status(path)
            if route == "/status.json":
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", payload)
            elif route in {"/healthz", "/ping"}:
                payload = json.dumps(
                    {"state": data.get("state", "initializing")}
                ).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", payload)
            else:
                payload = render_page(data).encode("utf-8")
                self._send(200, "text/html; charset=utf-8", payload)

        def log_message(self, _format, *_args):
            return

    server = ReusableServer((host, port), Handler)
    print(f"BOOT STATUS: listening on http://{host}:{port}", flush=True)

    def stop(_signum, _frame):
        server.server_close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write")
    write.add_argument("--file", required=True)
    write.add_argument("--state")
    write.add_argument("--phase")
    write.add_argument("--message")
    write.add_argument("--detail")
    write.add_argument("--assets-completed", type=int)
    write.add_argument("--assets-total", type=int)
    write.add_argument("--bytes-completed", type=int)
    write.add_argument("--bytes-total", type=int)

    server = subparsers.add_parser("serve")
    server.add_argument("--file", required=True)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8188)

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.file, args.host, args.port)
        return 0

    write_status(
        args.file,
        {
            "state": args.state,
            "phase": args.phase,
            "message": args.message,
            "detail": args.detail,
            "assets_completed": args.assets_completed,
            "assets_total": args.assets_total,
            "bytes_completed": args.bytes_completed,
            "bytes_total": args.bytes_total,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
