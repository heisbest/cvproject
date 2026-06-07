"""
Web demo: upload an image, visualize detection boxes (xywh) and per-class probabilities.

Usage:
  python web_demo.py --weights-dir ./weights/animal-90
  python web_demo.py --data-path ./data/animal-90   # infer weights dir
  python web_demo.py --weights-dir ./weights/animal-90 --port 7860
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.pipeline import AnimalRecognitionPipeline
from utils.paths import weights_dir_for_dataset

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>动物识别 · 定位 + 分类</title>
  <style>
    :root { --bg:#0f1419; --card:#1a2332; --accent:#00e676; --text:#e8eef5; --muted:#8b9cb3; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    header { padding:1.2rem 2rem; border-bottom:1px solid #2a3544; }
    header h1 { margin:0; font-size:1.35rem; font-weight:600; }
    header p { margin:.35rem 0 0; color:var(--muted); font-size:.9rem; }
    main { max-width:1200px; margin:0 auto; padding:1.5rem; display:grid; gap:1.25rem; grid-template-columns:1fr 340px; }
    @media(max-width:900px){ main{ grid-template-columns:1fr; } }
    .card { background:var(--card); border-radius:12px; padding:1.25rem; border:1px solid #2a3544; }
    .upload { border:2px dashed #3d4f66; border-radius:10px; padding:2rem; text-align:center; cursor:pointer; transition:.2s; }
    .upload:hover { border-color:var(--accent); background:#15202b; }
    .upload input { display:none; }
    #canvas-wrap { position:relative; display:inline-block; max-width:100%; margin-top:1rem; }
    #preview { max-width:100%; border-radius:8px; display:block; }
    #overlay { position:absolute; left:0; top:0; pointer-events:none; }
    .box-label { position:absolute; background:rgba(0,230,118,.9); color:#000; font-size:12px; font-weight:600; padding:2px 6px; border-radius:4px; white-space:nowrap; }
    .det { border:1px solid #2a3544; border-radius:8px; padding:.75rem; margin-bottom:.75rem; cursor:pointer; transition:.15s; }
    .det:hover, .det.active { border-color:var(--accent); background:#15202b; }
    .det h3 { margin:0 0 .5rem; font-size:.95rem; color:var(--accent); }
    .prob-row { display:flex; align-items:center; gap:.5rem; margin:.25rem 0; font-size:.78rem; }
    .prob-row span { width:72px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .bar { flex:1; height:6px; background:#2a3544; border-radius:3px; overflow:hidden; }
    .bar i { display:block; height:100%; background:var(--accent); border-radius:3px; }
    .prob-row b { width:42px; text-align:right; font-size:.75rem; }
    button { background:var(--accent); color:#000; border:none; padding:.65rem 1.2rem; border-radius:8px; font-weight:600; cursor:pointer; margin-top:.75rem; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .status { color:var(--muted); font-size:.85rem; margin-top:.5rem; }
    .xywh { font-family:monospace; font-size:.75rem; color:var(--muted); margin-bottom:.35rem; }
  </style>
</head>
<body>
  <header>
    <h1>动物图像识别 · Mask-First + CBAM</h1>
    <p>上传图片 → 显示每个检测框 (xywh) 及框内动物类别概率</p>
  </header>
  <main>
    <section class="card">
      <label class="upload" id="drop">
        <input type="file" id="file" accept="image/*"/>
        <div>点击或拖拽上传图片</div>
      </label>
      <button id="run" disabled>开始识别</button>
      <div class="status" id="status"></div>
      <div id="canvas-wrap" style="display:none">
        <img id="preview" alt="preview"/>
        <canvas id="overlay"></canvas>
      </div>
    </section>
    <aside class="card">
      <h2 style="margin:0 0 1rem;font-size:1rem">检测结果</h2>
      <div id="results"><p class="status">等待上传…</p></div>
    </aside>
  </main>
  <script>
    const fileInput = document.getElementById('file');
    const drop = document.getElementById('drop');
    const runBtn = document.getElementById('run');
    const preview = document.getElementById('preview');
    const overlay = document.getElementById('overlay');
    const wrap = document.getElementById('canvas-wrap');
    const resultsEl = document.getElementById('results');
    const statusEl = document.getElementById('status');
    let currentFile = null;
    let lastDetections = [];

    drop.onclick = () => fileInput.click();
    drop.ondragover = e => { e.preventDefault(); drop.style.borderColor = '#00e676'; };
    drop.ondragleave = () => drop.style.borderColor = '';
    drop.ondrop = e => { e.preventDefault(); drop.style.borderColor = ''; if(e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); };
    fileInput.onchange = () => { if(fileInput.files[0]) setFile(fileInput.files[0]); };

    function setFile(f) {
      currentFile = f;
      runBtn.disabled = false;
      statusEl.textContent = '已选择: ' + f.name;
      const url = URL.createObjectURL(f);
      preview.onload = () => { wrap.style.display = 'inline-block'; drawBoxes([]); };
      preview.src = url;
    }

    function drawBoxes(dets, activeIdx = -1) {
      const w = preview.clientWidth, h = preview.clientHeight;
      const nw = preview.naturalWidth, nh = preview.naturalHeight;
      overlay.width = w; overlay.height = h;
      overlay.style.width = w + 'px'; overlay.style.height = h + 'px';
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, w, h);
      dets.forEach((d, i) => {
        const [x,y,bw,bh] = d.bbox_xywh;
        const x1 = x*w, y1 = y*h, x2 = (x+bw)*w, y2 = (y+bh)*h;
        ctx.strokeStyle = i === activeIdx ? '#ffeb3b' : '#00e676';
        ctx.lineWidth = i === activeIdx ? 3 : 2;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
        ctx.fillStyle = i === activeIdx ? 'rgba(255,235,59,.15)' : 'rgba(0,230,118,.12)';
        ctx.fillRect(x1, y1, x2-x1, y2-y1);
      });
    }

    function renderResults(dets) {
      lastDetections = dets;
      if (!dets.length) { resultsEl.innerHTML = '<p class="status">未检测到动物</p>'; return; }
      resultsEl.innerHTML = dets.map((d,i) => {
        const top = (d.top5 || []).map(([name,p]) =>
          `<div class="prob-row"><span>${name}</span><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div><b>${(p*100).toFixed(1)}%</b></div>`
        ).join('');
        const [x,y,w,h] = d.bbox_xywh;
        return `<div class="det" data-i="${i}"><div class="xywh">框 ${i+1} · xywh (${x.toFixed(3)}, ${y.toFixed(3)}, ${w.toFixed(3)}, ${h.toFixed(3)})</div>
          <h3>${d.class} · ${(d.confidence*100).toFixed(1)}%</h3>${top}</div>`;
      }).join('');
      document.querySelectorAll('.det').forEach(el => {
        el.onclick = () => {
          document.querySelectorAll('.det').forEach(x => x.classList.remove('active'));
          el.classList.add('active');
          drawBoxes(lastDetections, +el.dataset.i);
        };
      });
    }

    runBtn.onclick = async () => {
      if (!currentFile) return;
      runBtn.disabled = true;
      statusEl.textContent = '识别中…';
      const fd = new FormData();
      fd.append('image', currentFile);
      try {
        const res = await fetch('/predict', { method:'POST', body: fd });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderResults(data.detections || []);
        statusEl.textContent = `完成 · ${(data.detections||[]).length} 个检测框`;
        drawBoxes(data.detections || []);
      } catch(e) {
        statusEl.textContent = '错误: ' + e.message;
      }
      runBtn.disabled = false;
    };
  </script>
</body>
</html>"""


def create_app(pipeline: AnimalRecognitionPipeline) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "no image uploaded"}), 400

        f = request.files["image"]
        try:
            image = Image.open(f.stream).convert("RGB")
            detections = pipeline.predict_image(image)
            return jsonify({"detections": detections})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="Web demo for animal recognition")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-locateanything", action="store_true")
    args = parser.parse_args()

    if args.weights_dir:
        weights_dir = Path(args.weights_dir)
    elif args.data_path:
        weights_dir = weights_dir_for_dataset(args.data_path)
    else:
        parser.error("Provide --weights-dir or --data-path")

    if not (weights_dir / "classifier_cbam_best.pth").exists():
        print(f"Warning: no trained classifier in {weights_dir}. Train first:")
        print(f"  python scripts/train.py --data-path <your-dataset>")

    pipeline = AnimalRecognitionPipeline(
        weights_dir,
        use_locateanything=not args.no_locateanything,
    )
    app = create_app(pipeline)
    print(f"Weights: {weights_dir}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
