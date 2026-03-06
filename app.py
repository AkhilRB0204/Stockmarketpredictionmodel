
import os, time, threading
from datetime import datetime
from collections import deque

from flask import Flask, Response, jsonify, render_template_string
import numpy as np

from data       import get_live_data, add_features, EXCLUDE_COLS
from model      import train_model, save_model
from validation import ModelValidator

# Config
TICKER               = os.getenv("TICKER", "AAPL")
RETRAIN_EVERY        = int(os.getenv("RETRAIN_EVERY",        "30"))
MIN_RETRAIN_INTERVAL = int(os.getenv("MIN_RETRAIN_INTERVAL", "10"))
POLL_INTERVAL        = int(os.getenv("POLL_INTERVAL",        "60"))
HISTORY_LEN          = 200
ANOMALY_WINDOW       = 20
OUTPUT_DIR           = "/app/output"
MODEL_PATH           = os.path.join(OUTPUT_DIR, "model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# Shared state  (all writes inside `lock`)
lock  = threading.Lock()
state = {
    "timestamps":    deque(maxlen=HISTORY_LEN),
    "actual":        deque(maxlen=HISTORY_LEN),
    "predicted":     deque(maxlen=HISTORY_LEN),
    "upper_ci":      deque(maxlen=HISTORY_LEN),
    "lower_ci":      deque(maxlen=HISTORY_LEN),
    "anomalies":     deque(maxlen=HISTORY_LEN),
    "signals":       deque(maxlen=60),
    "model":         None,
    "mae":           None,
    "rmse":          None,
    "direction_acc": None,
    "retrain_count": 0,
    "counter":       0,
    "last_retrain":  0,
    "status":        "Initialising…",
    "last_update":   None,
}

# Anomaly detection

def detect_anomalies(actual_list: list, live_row: dict):
    """
    Five-signal anomaly detector.
    Returns (is_anomaly: bool, reasons: list[str])
    """
    reasons = []

    # 1 · Price z-score
    prices = np.array(actual_list[-ANOMALY_WINDOW:])
    if len(prices) >= 5:
        mu, sigma = prices.mean(), prices.std()
        if sigma > 0:
            z = abs((prices[-1] - mu) / sigma)
            if z > 2.5:
                reasons.append(f"Price z-score {z:.1f}σ")

    # 2 · Volatility spike
    vol = float(live_row.get("Volatility", 0))
    if vol > 0.003:
        reasons.append(f"Volatility spike {vol * 100:.3f}%")

    # 3 · RSI extremes
    rsi = float(live_row.get("RSI", 50))
    if rsi > 75:
        reasons.append(f"RSI overbought {rsi:.0f}")
    elif rsi < 25:
        reasons.append(f"RSI oversold {rsi:.0f}")

    # 4 · Volume surge
    vr = float(live_row.get("Volume_Ratio", 1))
    if vr > 3:
        reasons.append(f"Volume surge ×{vr:.1f}")

    # 5 · MACD crossover
    macd     = float(live_row.get("MACD", 0))
    macd_sig = float(live_row.get("MACD_Signal", 0))
    if abs(macd) > 0 and abs(macd - macd_sig) < 0.01:
        reasons.append("MACD crossover")

    return bool(reasons), reasons

# Background worker thread

def _signal(ts: str, kind: str, msg: str):
    """Prepend a signal entry — must be called inside lock."""
    state["signals"].appendleft({"time": ts, "type": kind, "msg": msg})


def worker():
    validator = ModelValidator()
    with lock:
        state["status"] = "Fetching historical data…"
    try:
        hist  = add_features(get_live_data(TICKER))
        model = train_model(hist)
        save_model(model, MODEL_PATH)

        # Log top features
        importance = model.get_feature_importance(model.feature_names)
        top = list(importance.items())[:5]
        top_str = "  ".join(f"{f}={v:.3f}" for f, v in top)

        with lock:
            state["model"]  = model
            state["status"] = "Live"
            _signal(datetime.now().strftime("%H:%M:%S"), "info",
                    f"Model trained · {len(hist)} bars")
            _signal(datetime.now().strftime("%H:%M:%S"), "info",
                    f"Top features: {top_str}")
    except Exception as e:
        with lock:
            state["status"] = f"Init error: {e}"
        return

    last_ts = None

    # Main polling loop 
    while True:
        try:
            live_data  = add_features(get_live_data(TICKER))
            current_ts = live_data.index[-1]

            if current_ts == last_ts:
                time.sleep(30)
                continue
            last_ts = current_ts

            # Predict
            feat_cols  = [c for c in live_data.columns if c not in EXCLUDE_COLS]
            latest     = live_data[feat_cols].iloc[-1].to_numpy(dtype=float).reshape(1, -1)
            price      = float(live_data["Close"].iloc[-1])
            volatility = float(live_data["Volatility"].iloc[-1])

            with lock:
                model = state["model"]

            pred_return = float(model.predict(latest)[0])
            pred_price  = price * (1 + pred_return)
            ci          = price * volatility * 1.96  # 95% CI

            # Anomaly detection
            with lock:
                actual_hist = list(state["actual"])
            live_row    = {k: float(v) for k, v in live_data.iloc[-1].items()
                           if isinstance(v, (int, float, np.floating))}
            is_anomaly, reasons = detect_anomalies(actual_hist, live_row)

            # Rolling performance metrics
            with lock:
                preds   = list(state["predicted"])
                actuals = list(state["actual"])

            mae = rmse = da = None
            if len(actuals) > 5:
                n     = min(20, len(actuals) - 1)
                a_arr = np.array(actuals[-n:])
                p_arr = np.array(preds[-n:])
                mae   = float(np.mean(np.abs(a_arr - p_arr)))
                rmse  = float(np.sqrt(np.mean((a_arr - p_arr) ** 2)))
                da    = float(np.mean(
                            np.sign(np.diff(a_arr)) == np.sign(np.diff(p_arr))
                        ) * 100)

            # Update validator
            if len(actuals) > 0:
                validator.add_prediction(
                    timestamp=current_ts,
                    predicted=preds[-1] if preds else pred_price,
                    actual=price,
                )

            ts_str = current_ts.strftime("%H:%M")

            # Write shared state
            with lock:
                state["timestamps"].append(ts_str)
                state["actual"].append(price)
                state["predicted"].append(pred_price)
                state["upper_ci"].append(pred_price + ci)
                state["lower_ci"].append(pred_price - ci)
                state["anomalies"].append(is_anomaly)
                state["counter"]      += 1
                state["mae"]           = round(mae,  4) if mae  is not None else None
                state["rmse"]          = round(rmse, 4) if rmse is not None else None
                state["direction_acc"] = round(da,   1) if da   is not None else None
                state["last_update"]   = datetime.now().strftime("%H:%M:%S")
                if is_anomaly:
                    for r in reasons:
                        _signal(ts_str, "anomaly", f"⚠  {r}")

            # Retraining
            with lock:
                counter      = state["counter"]
                last_retrain = state["last_retrain"]

            should_retrain, retrain_reason = False, ""
            if counter % RETRAIN_EVERY == 0:
                should_retrain  = True
                retrain_reason  = "scheduled"
            elif (counter - last_retrain >= MIN_RETRAIN_INTERVAL
                  and validator.should_retrain()):
                should_retrain  = True
                retrain_reason  = f"MAE ${mae:.4f} > threshold"

            if should_retrain:
                with lock:
                    state["status"] = "Retraining…"
                new_hist  = add_features(get_live_data(TICKER))
                new_model = train_model(new_hist)
                save_model(new_model, MODEL_PATH)
                with lock:
                    state["model"]         = new_model
                    state["last_retrain"]  = counter
                    state["retrain_count"] += 1
                    state["status"]        = "Live"
                    _signal(ts_str, "retrain",
                            f"🔄 Retrained #{state['retrain_count']} ({retrain_reason})")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            with lock:
                _signal(datetime.now().strftime("%H:%M:%S"), "error", f"✖ {e}")
            time.sleep(POLL_INTERVAL)

# Dashboard HTML  (defined BEFORE routes so index() can reference it)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ ticker }} · Live ML Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:     #0a0e1a; --card:  #0f1424; --border: #1e2540;
  --dim:    #4a5278; --muted: #7a82a8; --bright: #e8eaf6;
  --blue:   #4078ff; --amber: #f59e0b; --green:  #34d399;
  --red:    #f87171; --purple:#a78bfa;
  --mono: 'Space Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--bright);font-family:var(--sans);font-size:14px}
body{display:grid;grid-template-rows:52px 1fr;grid-template-columns:1fr 300px;
     grid-template-areas:"nav nav""main side";min-height:100vh}

nav{grid-area:nav;background:var(--card);border-bottom:1px solid var(--border);
    display:flex;align-items:center;padding:0 1.5rem;gap:1.25rem;
    position:sticky;top:0;z-index:10}
.nav-ticker{font-family:var(--mono);font-size:.7rem;letter-spacing:.15em;color:var(--blue);
            padding:.22rem .55rem;border:1px solid var(--blue);border-radius:4px}
.nav-price{font-family:var(--mono);font-size:1.05rem;color:var(--bright)}
.nav-change{font-family:var(--mono);font-size:.78rem}
.nav-change.up{color:var(--green)}.nav-change.down{color:var(--red)}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:.75rem}
.status-pill{display:flex;align-items:center;gap:.4rem;font-family:var(--mono);
             font-size:.65rem;color:var(--dim)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex-shrink:0}
.dot.live{background:var(--green);box-shadow:0 0 6px var(--green);animation:blink 2s infinite}
.dot.retrain{background:var(--amber);box-shadow:0 0 6px var(--amber)}
.dot.error{background:var(--red);box-shadow:0 0 6px var(--red)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}

main{grid-area:main;padding:1.1rem;display:flex;flex-direction:column;gap:.9rem;overflow:auto}
aside{grid-area:side;background:var(--card);border-left:1px solid var(--border);
      padding:1rem;display:flex;flex-direction:column;gap:1rem;overflow-y:auto}

.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem}
.card-title{font-family:var(--mono);font-size:.6rem;letter-spacing:.13em;text-transform:uppercase;
            color:var(--dim);margin-bottom:.7rem}

.stats-row{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem}
.stat{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.7rem .9rem}
.stat label{display:block;font-family:var(--mono);font-size:.58rem;letter-spacing:.1em;
            text-transform:uppercase;color:var(--dim);margin-bottom:.28rem}
.stat .val{font-family:var(--mono);font-size:.98rem;color:var(--bright)}
.stat .val.up{color:var(--green)}.stat .val.down{color:var(--red)}.stat .val.warn{color:var(--amber)}

.chart-wrap{position:relative;height:320px}
.chart-wrap-sm{position:relative;height:130px}
.anomaly-badge{position:absolute;top:8px;right:8px;background:rgba(245,158,11,.13);
               border:1px solid var(--amber);color:var(--amber);font-family:var(--mono);
               font-size:.6rem;padding:.18rem .5rem;border-radius:4px;display:none}
.anomaly-badge.on{display:block}

.signal-list{display:flex;flex-direction:column;gap:.35rem;max-height:240px;overflow-y:auto}
.signal{display:flex;gap:.55rem;padding:.38rem .45rem;border-radius:5px;font-size:.68rem;
        font-family:var(--mono);border-left:2px solid transparent}
.signal.info{border-color:var(--blue);background:rgba(64,120,255,.05)}
.signal.anomaly{border-color:var(--amber);background:rgba(245,158,11,.06)}
.signal.retrain{border-color:var(--purple);background:rgba(167,139,250,.05)}
.signal.error{border-color:var(--red);background:rgba(248,113,113,.05)}
.sig-time{color:var(--dim);white-space:nowrap;flex-shrink:0}
.sig-msg{color:var(--muted);line-height:1.45}

.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;
           font-family:var(--mono);font-size:.64rem;color:var(--muted);line-height:1.8}
.info-grid span{color:var(--dim)}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>

<nav>
  <span class="nav-ticker">{{ ticker }}</span>
  <span class="nav-price"  id="nav-price">—</span>
  <span class="nav-change" id="nav-change"></span>
  <span style="font-family:var(--mono);font-size:.65rem;color:var(--dim)">yfinance · 1 min</span>
  <div class="nav-right">
    <div class="status-pill">
      <div class="dot" id="status-dot"></div>
      <span id="status-text">Initialising…</span>
    </div>
    <span id="last-update" style="font-family:var(--mono);font-size:.62rem;color:var(--dim)"></span>
  </div>
</nav>

<main>
  <div class="stats-row">
    <div class="stat"><label>Last Price</label><div class="val" id="s-price">—</div></div>
    <div class="stat"><label>Predicted</label> <div class="val" id="s-pred">—</div></div>
    <div class="stat"><label>MAE (20)</label>  <div class="val" id="s-mae">—</div></div>
    <div class="stat"><label>Dir. Acc.</label> <div class="val" id="s-da">—</div></div>
    <div class="stat"><label>Retrains</label>  <div class="val" id="s-rt">0</div></div>
  </div>

  <div class="card" style="flex:1;position:relative">
    <div class="card-title">Actual vs Predicted · 95% Confidence Interval</div>
    <div class="anomaly-badge" id="anomaly-badge">⚠ ANOMALY</div>
    <div class="chart-wrap"><canvas id="price-chart"></canvas></div>
  </div>

  <div class="card">
    <div class="card-title">Prediction Error (actual − predicted₋₁)</div>
    <div class="chart-wrap-sm"><canvas id="error-chart"></canvas></div>
  </div>
</main>

<aside>
  <div>
    <div class="card-title">Signal Feed</div>
    <div class="signal-list" id="signal-list">
      <div class="signal info">
        <span class="sig-time">--:--</span>
        <span class="sig-msg">Waiting for data…</span>
      </div>
    </div>
  </div>
  <div>
    <div class="card-title">Ensemble Model</div>
    <div class="info-grid">
      <span>RandomForest</span> 50%
      <span>Grad. Boost</span>  30%
      <span>Ridge</span>        20%
      <span>Features</span>     SelectK 15
      <span>Target</span>       next return
      <span>Retrain</span>      every {{ retrain_every }}
    </div>
  </div>
  <div>
    <div class="card-title">Anomaly Rules</div>
    <div class="info-grid">
      <span>Z-score</span>   &gt; 2.5σ
      <span>Volatility</span>&gt; 0.3%
      <span>RSI hi</span>    &gt; 75
      <span>RSI lo</span>    &lt; 25
      <span>Volume</span>    &gt; 3×
      <span>MACD</span>      crossover
    </div>
  </div>
</aside>

<script>
const C={bg:'#0f1424',border:'#1e2540',dim:'#4a5278',muted:'#7a82a8',
         bright:'#e8eaf6',blue:'#4078ff',amber:'#f59e0b',green:'#34d399',red:'#f87171'};

function mkScales(yFmt){
  return{
    x:{grid:{color:'rgba(255,255,255,0.025)',drawTicks:false},border:{display:false},
       ticks:{color:C.dim,font:{family:'Space Mono',size:8},maxRotation:0,maxTicksLimit:10}},
    y:{grid:{color:'rgba(255,255,255,0.04)',drawTicks:false},border:{display:false},
       position:'right',ticks:{color:C.dim,font:{family:'Space Mono',size:8},callback:yFmt}}
  };
}
function mkTooltip(labelFn){
  return{backgroundColor:'#141828',titleColor:C.dim,bodyColor:C.bright,
         borderColor:C.border,borderWidth:1,padding:10,
         titleFont:{family:'Space Mono',size:9},bodyFont:{family:'Space Mono',size:10},
         callbacks:{label:labelFn}};
}

const pCtx=document.getElementById('price-chart').getContext('2d');
const priceChart=new Chart(pCtx,{
  type:'line',
  data:{labels:[],datasets:[
    {data:[],fill:'+1',backgroundColor:'rgba(245,158,11,0.07)',borderWidth:0,pointRadius:0,tension:.35},
    {data:[],fill:false,borderWidth:0,pointRadius:0,tension:.35},
    {label:'Actual',data:[],borderColor:C.blue,backgroundColor:'rgba(64,120,255,0.06)',
     fill:true,borderWidth:2,tension:.35,pointRadius:0,pointHoverRadius:5,pointBackgroundColor:C.blue},
    {label:'Predicted',data:[],borderColor:C.amber,borderDash:[6,4],
     backgroundColor:'transparent',borderWidth:1.5,tension:.35,pointRadius:0,pointHoverRadius:4},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:350},
           interaction:{mode:'index',intersect:false},
           plugins:{legend:{display:false},
                    tooltip:mkTooltip(item=>{
                      if(item.datasetIndex<2)return null;
                      const l=item.datasetIndex===2?'Actual   ':'Predicted';
                      return item.raw!=null?`${l}  $${Number(item.raw).toFixed(2)}`:null;
                    })},
           scales:mkScales(v=>`$${v.toFixed(2)}`)}
});

const eCtx=document.getElementById('error-chart').getContext('2d');
const errChart=new Chart(eCtx,{
  type:'bar',
  data:{labels:[],datasets:[{label:'Error',data:[],backgroundColor:[],borderWidth:0,borderRadius:3}]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:200},
           plugins:{legend:{display:false},
                    tooltip:mkTooltip(item=>`Error  $${Number(item.raw)>=0?'+':''}${Number(item.raw).toFixed(4)}`)},
           scales:mkScales(v=>`$${v>=0?'+':''}${v.toFixed(3)}`)}
});

let prevPrice=null;

async function refresh(){
  let d;
  try{d=await fetch('/api/data').then(r=>r.json());}catch{return;}

  const price=d.actual.at(-1), pred=d.predicted.at(-1);

  if(price!=null){
    document.getElementById('nav-price').textContent=`$${price.toFixed(2)}`;
    if(prevPrice!=null){
      const chg=(price-prevPrice)/prevPrice*100;
      const el=document.getElementById('nav-change');
      el.textContent=`${chg>=0?'▲':'▼'} ${Math.abs(chg).toFixed(3)}%`;
      el.className='nav-change '+(chg>=0?'up':'down');
    }
    prevPrice=price;
  }

  const dot=document.getElementById('status-dot');
  dot.className='dot '+(d.status==='Live'?'live':d.status.includes('Retrain')?'retrain':'error');
  document.getElementById('status-text').textContent=d.status;
  if(d.last_update)
    document.getElementById('last-update').textContent=`updated ${d.last_update}`;

  document.getElementById('s-price').textContent=price!=null?`$${price.toFixed(2)}`:'—';
  document.getElementById('s-pred').textContent =pred!=null ?`$${pred.toFixed(2)}` :'—';
  document.getElementById('s-rt').textContent   =d.retrain_count??0;

  const maeEl=document.getElementById('s-mae');
  maeEl.textContent=d.mae!=null?`$${d.mae}`:'—';
  maeEl.className='val'+(d.mae!=null&&d.mae>0.3?' warn':'');

  const daEl=document.getElementById('s-da');
  daEl.textContent=d.direction_acc!=null?`${d.direction_acc}%`:'—';
  daEl.className='val '+(d.direction_acc>=55?'up':d.direction_acc<45?'down':'');

  document.getElementById('anomaly-badge').className=
    'anomaly-badge'+(d.anomalies.at(-1)?' on':'');

  priceChart.data.labels=d.timestamps;
  priceChart.data.datasets[0].data=d.upper_ci;
  priceChart.data.datasets[1].data=d.lower_ci;
  priceChart.data.datasets[2].data=d.actual;
  priceChart.data.datasets[3].data=d.predicted;
  priceChart.data.datasets[2].pointRadius=d.anomalies.map(a=>a?5:0);
  priceChart.data.datasets[2].pointBackgroundColor=d.anomalies.map(a=>a?C.amber:C.blue);
  priceChart.update('none');

  const errors=d.actual.map((a,i)=>i===0?0:parseFloat((a-d.predicted[i-1]).toFixed(4)));
  errChart.data.labels=d.timestamps;
  errChart.data.datasets[0].data=errors;
  errChart.data.datasets[0].backgroundColor=errors.map(e=>e>=0?'rgba(52,211,153,0.55)':'rgba(248,113,113,0.55)');
  errChart.update('none');

  const feed=document.getElementById('signal-list');
  if(d.signals.length)
    feed.innerHTML=d.signals.map(s=>
      `<div class="signal ${s.type}">
         <span class="sig-time">${s.time}</span>
         <span class="sig-msg">${s.msg}</span>
       </div>`).join('');
}

refresh();
setInterval(refresh,30_000);
const es=new EventSource('/api/stream');
es.onmessage=()=>refresh();
es.onerror=()=>{};
</script>
</body>
</html>"""

# Flask routes

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML, ticker=TICKER,
                                  retrain_every=RETRAIN_EVERY)


@app.route("/api/data")
def api_data():
    with lock:
        return jsonify({
            "timestamps":    list(state["timestamps"]),
            "actual":        list(state["actual"]),
            "predicted":     list(state["predicted"]),
            "upper_ci":      list(state["upper_ci"]),
            "lower_ci":      list(state["lower_ci"]),
            "anomalies":     list(state["anomalies"]),
            "signals":       list(state["signals"])[:20],
            "status":        state["status"],
            "mae":           state["mae"],
            "rmse":          state["rmse"],
            "direction_acc": state["direction_acc"],
            "retrain_count": state["retrain_count"],
            "last_update":   state["last_update"],
        })


@app.route("/api/stream")
def api_stream():
    """SSE — pushes a ping every 30 s so the browser refreshes the chart."""
    def generate():
        while True:
            yield "data: ping\n\n"
            time.sleep(30)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/health")
def health():
    """Docker HEALTHCHECK endpoint."""
    with lock:
        return jsonify({"status": state["status"],
                        "last_update": state["last_update"]}), 200

# Entry point

if __name__ == "__main__":
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)