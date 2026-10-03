"""Static HTML 'storm radar': hex map of forecast storm probability stepping through a surge event.

Every forecast shown comes from a model that never saw that cell's district (leave-one-district-out predictions)."""
from __future__ import annotations

import json
import math

import numpy as np
import polars as pl

from .config import Config
from .sim import build_layout
from .store import results_dir

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tailwatch storm radar</title><style>
:root{--bg:#0f1720;--fg:#e6edf3;--mut:#8b98a5;--card:#18222d}
@media (prefers-color-scheme: light){:root{--bg:#f6f8fa;--fg:#1f2328;--mut:#59636e;--card:#fff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px}h1{margin:0 0 4px;font-size:22px}p{color:var(--mut);margin:4px 0 14px}
.card{background:var(--card);border-radius:10px;padding:14px;box-shadow:0 1px 3px #0003}
canvas{width:100%;height:auto;display:block}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:10px}
input[type=range]{flex:1;min-width:200px}button{padding:6px 14px;border-radius:6px;border:0;background:#2f81f7;color:#fff;cursor:pointer}
.legend span{display:inline-block;margin-right:14px;color:var(--mut);font-size:13px}.sw{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin-right:4px;border-radius:2px}
</style></head><body><main>
<h1>tailwatch storm radar</h1>
<p>Forecast probability that a cell hits congestion within __H__ s. Synthetic city digital twin, not real operator data.
Each forecast comes from a model trained <b>without</b> that cell's district.</p>
<div class="card"><canvas id="c" width="900" height="__CH__"></canvas>
<div class="row"><button id="play">Play</button><input id="s" type="range" min="0" max="__N__" value="0"><span id="clock"></span></div>
<div class="legend" style="margin-top:8px"><span><i class="sw" style="background:#2c7bb6"></i>low risk</span><span><i class="sw" style="background:#fdae61"></i>elevated</span><span><i class="sw" style="background:#d7191c"></i>high</span><span><i class="sw" style="background:#222;border:1px solid #888"></i>already congested</span><span>white ring = congestion actually began within the horizon</span></div></div>
</main><script>
const D=__DATA__;const cv=document.getElementById('c'),g=cv.getContext('2d'),sl=document.getElementById('s'),ck=document.getElementById('clock');
const S=D.size;function col(p){const a=Math.max(0,Math.min(1,p/D.pmax));const st=[[44,123,182],[253,174,97],[215,25,28]];
const x=a*2,i=Math.min(1,Math.floor(x)),f=x-i;return 'rgb('+st[i].map((v,k)=>Math.round(v+(st[i+1][k]-v)*f)).join(',')+')';}
function hex(x,y,r){g.beginPath();for(let k=0;k<6;k++){const a=Math.PI/180*(60*k-30);g.lineTo(x+r*Math.cos(a),y+r*Math.sin(a));}g.closePath();}
function draw(){const f=D.frames[+sl.value];g.clearRect(0,0,cv.width,cv.height);g.font='12px system-ui';
D.cells.forEach((c,i)=>{const x=S*Math.sqrt(3)*(c.col+0.5*(c.row%2))+S*1.2,y=S*1.5*c.row+S*1.2;const v=f[i];
hex(x,y,S*0.95);g.fillStyle=v.s==2?'#222':col(v.p);g.fill();if(v.s==1){g.lineWidth=3;g.strokeStyle='#fff';g.stroke();}
if(c.col==D.label_cols[c.district]&&c.row==0){g.fillStyle='#8b98a5';g.fillText(c.district,x-S*0.7,y-S*1.1);}});
const ev=D.event;const t=D.t0+ +sl.value*D.step;ck.textContent='t = '+t+' s'+(t>=ev.start&&t<ev.end?'  |  '+ev.district+' surge in progress':'');}
sl.oninput=draw;let timer=null;document.getElementById('play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='Play';}
else{this.textContent='Pause';timer=setInterval(()=>{sl.value=(+sl.value+1)%(+sl.max+1);draw();},120);}};draw();
</script></body></html>"""


def build(df: pl.DataFrame, oof: pl.DataFrame, cfg: Config) -> str:
    layout = build_layout(cfg)
    ev = cfg.sim.events[cfg.radar.event_index]
    t0 = max(0, ev.start_s - cfg.radar.lead_s)
    t0 -= t0 % cfg.features.stride_s
    t1 = min(cfg.sim.duration_s, t0 + cfg.radar.window_s)
    step = cfg.features.stride_s
    times = list(range(t0, t1, step))
    test = oof.filter((pl.col("strategy") == cfg.radar.strategy) & (pl.col("split") == "test"))
    rows = test["row"].to_numpy()
    sub = df[rows].select("cell_id", "t").with_columns(
        p=pl.Series(test[f"p_{cfg.radar.variant}"]), y=pl.Series(test["y"]))
    sub = sub.filter((pl.col("t") >= t0) & (pl.col("t") < t1))
    lookup = {(r["cell_id"], r["t"]): (r["p"], r["y"]) for r in sub.iter_rows(named=True)}
    pmax = float(max(0.05, np.quantile(sub["p"].to_numpy(), 0.97))) if sub.height else 1.0
    frames = []
    for t in times:
        frame = []
        for cid in range(layout.n_cells):
            hit = lookup.get((cid, t))
            frame.append({"s": 2, "p": 0.0} if hit is None else {"s": int(hit[1]), "p": round(float(hit[0]), 4)})
        frames.append(frame)
    cells, label_cols = [], {}
    for cid in range(layout.n_cells):
        r, c = divmod(cid, layout.cols)
        d = layout.district_of_cell[cid]
        label_cols.setdefault(d, c)
        cells.append({"row": r, "col": c, "district": d})
    size = 900 / (math.sqrt(3) * (layout.cols + 0.5) + 2.4)
    data = {"cells": cells, "frames": frames, "t0": t0, "step": step, "pmax": pmax, "size": size,
            "label_cols": label_cols, "event": {"district": ev.district, "start": ev.start_s, "end": ev.start_s + ev.duration_s}}
    ch = int(size * (1.5 * layout.rows + 2.4))
    html = (TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__N__", str(len(times) - 1))
            .replace("__H__", str(cfg.storm.horizon_s)).replace("__CH__", str(ch)))
    path = results_dir(cfg) / "radar.html"
    path.write_text(html, encoding="utf-8")
    return str(path)
