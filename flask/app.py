from flask import Flask, render_template_string, request, jsonify, send_file
import pandas as pd
import os, json, random
from pathlib import Path
from glob import glob

app = Flask(__name__)

PATCHES_DIR      = "/Volumes/External DATA/Team1/MLGlom/patches"
LABELS_FILE      = "/Volumes/External DATA/Team1/MLGlom/labels/labels.csv"
PREDICTIONS_FILE = "/Volumes/External DATA/Team1/MLGlom/results/results_per_glom.csv"
VALIDATED_FILE   = "/Volumes/External DATA/Team1/MLGlom/labels/validated_labels.csv"
METADATA_FILE    = "/Volumes/External DATA/Team1/MLGlom/patches_metadata.json"

CLASSES = ["Normal","Adhesion","Thickening GBM","Fibrinoid necrosis",
           "Hypercellularity","Fibrosis","Crescent","Sclerosis",
           "Double glomerulus","Not a glom"]
SPECIAL = ["Double glomerulus","Not a glom"]

Path(VALIDATED_FILE).parent.mkdir(parents=True, exist_ok=True)

def load_predictions():
    """Load model predictions as initial classes."""
    if not os.path.exists(PREDICTIONS_FILE): return {}
    df = pd.read_csv(PREDICTIONS_FILE)
    result = {}
    for _, row in df.iterrows():
        lbl = str(row['classes']) if pd.notna(row['classes']) else ''
        if lbl and lbl != 'nan' and lbl != 'Unclassified':
            result[row['patch']] = lbl
    return result

def load_validated():
    """Load already validated labels."""
    if not os.path.exists(VALIDATED_FILE): return {}
    df = pd.read_csv(VALIDATED_FILE)
    result = {}
    for _, row in df.iterrows():
        lbl = str(row['labels']) if pd.notna(row['labels']) else ''
        if lbl and lbl != 'nan':
            result[row['patch']] = {
                'labels': lbl,
                'user':   str(row.get('user', '')),
                'status': str(row.get('status', 'validated'))
            }
    return result

def save_validated(patch, labels, user, original_pred):
    """Save validated label with status: confirmed or corrected."""
    data = load_validated()
    pred_set = set(original_pred.split('|')) if original_pred else set()
    new_set  = set(labels)
    status   = 'confirmed' if pred_set == new_set else 'corrected'
    data[patch] = {'labels': '|'.join(labels), 'user': user, 'status': status}
    rows = [{'patch':k,'labels':v['labels'],'user':v['user'],'status':v['status']}
            for k,v in data.items()]
    pd.DataFrame(rows).to_csv(VALIDATED_FILE, index=False)
    return status

def get_patches():
    p = sorted([Path(x).name for x in glob(os.path.join(PATCHES_DIR,"*.png"))])
    random.seed(42); random.shuffle(p); return p

def load_meta():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f: return json.load(f)
    return {}

HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Glomerulus Validator</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#eee}
#login-screen{position:fixed;inset:0;background:#1a1a2e;display:flex;align-items:center;justify-content:center;z-index:1000}
.login-box{background:#16213e;border:2px solid #0f3460;border-radius:16px;padding:40px;text-align:center;width:380px}
.login-box h1{color:#e94560;margin-bottom:8px;font-size:1.4em}
.login-box p{color:#aaa;margin-bottom:16px;font-size:.9em}
.login-box input{width:100%;padding:12px 16px;border-radius:8px;border:2px solid #0f3460;background:#1a1a2e;color:#eee;font-size:1em;margin-bottom:12px;outline:none}
.login-box input:focus{border-color:#e94560}
.login-box button{width:100%;padding:12px;background:#e94560;color:#fff;border:none;border-radius:8px;font-size:1em;font-weight:700;cursor:pointer}
header{background:#16213e;padding:12px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #0f3460}
header h1{font-size:1.1em;color:#e94560}
.hright{text-align:right}
.ubadge{font-size:.8em;color:#4caf50;margin-bottom:4px}
.stats{font-size:.82em;color:#aaa}
.pbar{height:5px;background:#0f3460;border-radius:3px;overflow:hidden;margin-top:4px;width:220px;margin-left:auto}
.pfill{height:100%;background:#e94560;transition:width .3s}
.main{display:flex;height:calc(100vh - 140px)}
.ipanel{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px;background:#16213e}
.cbox{position:relative;cursor:zoom-in;border-radius:8px;border:3px solid #0f3460;overflow:hidden}
#glom-img{display:block;max-width:100%;max-height:52vh;image-rendering:pixelated}
#overlay-canvas{position:absolute;top:0;left:0;pointer-events:none}
.pinfo{margin-top:10px;text-align:center}
.pname{font-size:.78em;color:#888}
.pidx{font-size:.72em;color:#555;margin-top:2px}
.pslide{font-size:.7em;color:#444;margin-top:1px}

/* Model prediction badge */
.pred-badge{
    margin-top:8px;
    padding:5px 12px;
    border-radius:12px;
    font-size:.75em;
    display:inline-block;
    border:1px solid #2196F3;
    color:#2196F3;
    background:rgba(33,150,243,.1);
}
.pred-badge.confirmed{border-color:#4caf50;color:#4caf50;background:rgba(76,175,80,.1)}
.pred-badge.corrected{border-color:#ff9800;color:#ff9800;background:rgba(255,152,0,.1)}

.navbtns{display:flex;gap:12px;margin-top:14px}
.nbtn{padding:10px 22px;border:none;background:#e94560;color:#fff;border-radius:8px;cursor:pointer;font-size:.9em;font-weight:700}
.nbtn:hover{background:#c73652}
.hints{margin-top:8px;font-size:.7em;color:#444;text-align:center}
.hints kbd{background:#0f3460;color:#aaa;border-radius:3px;padding:1px 5px}

.cpanel{width:320px;background:#16213e;border-left:2px solid #0f3460;padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:7px}
.cpanel h2{font-size:.9em;color:#888;margin-bottom:2px}
.cpanel .sub{font-size:.72em;color:#555;margin-bottom:6px}
.sep{height:1px;background:#0f3460;margin:4px 0}
.cbtn{display:flex;align-items:center;gap:10px;padding:9px 14px;border-radius:8px;border:2px solid #0f3460;background:#1a1a2e;color:#eee;cursor:pointer;font-size:.88em;transition:all .15s;user-select:none}
.cbtn:hover{border-color:#e94560;background:#2a1a2e}
.cbtn.sel{border-color:#e94560;background:#3d1a2e;color:#fff}
.cbtn.sp{border-color:#2a2a4e}
.cbtn.sp:hover{border-color:#ff9800;background:#2a1a0e}
.cbtn.sp.sel{border-color:#ff9800;background:#3d2a0e}
.cbtn.sp.sel .dot{background:#ff9800;border-color:#ff9800}
.cbtn.sp.sel .kn{color:#ff9800}
/* Model predicted class highlight */
.cbtn.predicted{border-color:#2196F3;background:rgba(33,150,243,.08)}
.cbtn.predicted.sel{border-color:#e94560;background:#3d1a2e}
.kn{font-size:.75em;color:#555;background:#0f3460;border-radius:3px;padding:1px 5px;min-width:20px;text-align:center}
.cbtn.sel .kn{color:#e94560}
.dot{width:14px;height:14px;border-radius:50%;border:2px solid #555;flex-shrink:0}
.cbtn.sel .dot{background:#e94560;border-color:#e94560}
.saved-msg{text-align:center;font-size:.8em;height:18px;margin-top:4px}
.saved-msg.ok{color:#4caf50}
.saved-msg.err{color:#e94560}

#zoom-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:500;align-items:center;justify-content:center;cursor:zoom-out}
#zoom-overlay.vis{display:flex}
#zoom-canvas{max-width:90vw;max-height:90vh;border-radius:8px}

#tstrip{position:fixed;bottom:0;left:0;right:0;height:70px;background:#0f3460;display:flex;overflow-x:auto;gap:3px;padding:4px;align-items:center;z-index:1}
#tstrip img{display:block;height:60px;width:60px;object-fit:cover;border-radius:4px;border:2px solid transparent;opacity:.5;cursor:pointer;flex-shrink:0}

/* Legend */
.legend{position:fixed;bottom:73px;right:8px;font-size:.66em;color:#555;text-align:right;line-height:1.9}
.ld{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:3px}
</style></head><body>

<div id="login-screen">
  <div class="login-box">
    <h1>&#128302; Glomerulus Validator</h1>
    <p>Model predictions are pre-checked.<br>Confirm or correct each glomerulus.</p>
    <input id="uname" type="text" placeholder="Your name..." maxlength="30">
    <button id="start-btn">Start validation &#8594;</button>
  </div>
</div>

<header>
  <h1>&#128302; Glomerulus Validator</h1>
  <div class="hright">
    <div class="ubadge" id="ubadge"></div>
    <div class="stats" id="stats">Loading...</div>
    <div class="pbar"><div class="pfill" id="pfill" style="width:0%"></div></div>
  </div>
</header>

<div class="main">
  <div class="ipanel">
    <div class="cbox" id="cbox">
      <img id="glom-img" src="" alt="">
      <canvas id="overlay-canvas"></canvas>
    </div>
    <div class="pinfo">
      <div class="pname"  id="pname"></div>
      <div class="pidx"   id="pidx"></div>
      <div class="pslide" id="pslide"></div>
      <div class="pred-badge" id="pred-badge">Model prediction</div>
    </div>
    <div class="navbtns">
      <button class="nbtn" id="prev-btn">&#8592; Save &amp; Previous</button>
      <button class="nbtn" id="next-btn">Save &amp; Next &#8594;</button>
    </div>
    <div class="hints">
      <kbd>&#8592;</kbd><kbd>&#8594;</kbd> save &amp; navigate &nbsp;
      <kbd>1</kbd>&#8211;<kbd>0</kbd> toggle &nbsp;
      <kbd>Space</kbd> zoom &nbsp;<kbd>Esc</kbd> close
    </div>
  </div>
  <div class="cpanel">
    <h2>Classification</h2>
    <div class="sub">&#128309; = model prediction &nbsp;|&nbsp; click to toggle</div>
    <div id="cbts"></div>
    <div class="saved-msg" id="saved-msg"></div>
  </div>
</div>

<div id="zoom-overlay"><canvas id="zoom-canvas"></canvas></div>
<div id="tstrip"></div>

<div class="legend">
  <span class="ld" style="background:#333"></span>unseen &nbsp;
  <span class="ld" style="background:#2196F3"></span>model pred &nbsp;
  <span class="ld" style="background:#4caf50"></span>confirmed &nbsp;
  <span class="ld" style="background:#ff9800"></span>corrected &nbsp;
  <span class="ld" style="background:#e94560"></span>active
</div>

<script>
var CLASSES     = {{ classes|tojson }};
var SPECIAL     = ["Double glomerulus","Not a glom"];
var patches     = {{ patches|tojson }};
var META        = {{ metadata|tojson }};
var PREDICTIONS = {{ predictions|tojson }};  // model predictions per patch
var VALIDATED   = {{ validated|tojson }};    // already validated

var curIdx = 0;
var sel    = [];
var user   = "";

// ── Helpers ───────────────────────────────────────────────────
function getPrediction(p) {
    var pred = PREDICTIONS[p] || "";
    if (pred === "Unclassified") return [];
    return pred ? pred.split("|").map(function(s){ return s.trim(); }).filter(Boolean) : [];
}

function getValidated(p) {
    var v = VALIDATED[p];
    return v ? v.labels.split("|").filter(Boolean) : null;
}

function getStatus(p) {
    var v = VALIDATED[p];
    if (!v) return "unseen";
    return v.status || "validated";
}

function hasLabel(p) {
    return !!VALIDATED[p];
}

// ── Login ─────────────────────────────────────────────────────
document.getElementById("start-btn").addEventListener("click", function() {
    var n = document.getElementById("uname").value.trim();
    if (!n) { alert("Please enter your name"); return; }
    user = n;
    document.getElementById("login-screen").style.display = "none";
    document.getElementById("ubadge").textContent = "&#128100; " + n;
    var si = 0;
    for (var i = 0; i < patches.length; i++) {
        if (!hasLabel(patches[i])) { si = i; break; }
    }
    loadPatch(si);
});
document.getElementById("uname").addEventListener("keydown", function(e) {
    if (e.key === "Enter") document.getElementById("start-btn").click();
});

document.getElementById("prev-btn").addEventListener("click", function() { doSaveNav(-1); });
document.getElementById("next-btn").addEventListener("click", function() { doSaveNav(1); });
document.getElementById("cbox").addEventListener("click", openZoom);
document.getElementById("zoom-overlay").addEventListener("click", closeZoom);

// ── Class buttons ─────────────────────────────────────────────
function renderCls(predClasses) {
    var c = document.getElementById("cbts");
    c.innerHTML = "";
    for (var i = 0; i < CLASSES.length; i++) {
        var cls   = CLASSES[i];
        var isSp  = SPECIAL.indexOf(cls) >= 0;
        var isSel = sel.indexOf(cls) >= 0;
        var isPred = predClasses.indexOf(cls) >= 0;

        if (cls === "Double glomerulus") {
            var s = document.createElement("div"); s.className = "sep"; c.appendChild(s);
        }

        var btn = document.createElement("div");
        btn.className = "cbtn" + (isSp?" sp":"") + (isSel?" sel":"") + (isPred&&!isSel?" predicted":"");
        btn.id = "cb"+i;
        var k  = i < 9 ? (i+1) : 0;
        // Show dot indicator for model prediction
        var predDot = isPred ? ' <span style="color:#2196F3;font-size:.7em">&#9679;</span>' : '';
        btn.innerHTML = '<div class="dot"></div><span style="flex:1">'+cls+predDot+'</span><span class="kn">'+k+'</span>';
        (function(c2,i2){ btn.addEventListener("click", function(){ toggleCls(c2,i2); }); })(cls,i);
        c.appendChild(btn);
    }
}

function toggleCls(cls, i) {
    var isSp = SPECIAL.indexOf(cls) >= 0;
    if (isSp && sel.indexOf(cls) < 0) {
        sel = [];
        document.querySelectorAll(".cbtn.sel").forEach(function(b){ b.classList.remove("sel"); });
    }
    if (!isSp) {
        for (var j = 0; j < SPECIAL.length; j++) {
            var idx2 = sel.indexOf(SPECIAL[j]);
            if (idx2 >= 0) {
                sel.splice(idx2,1);
                var sb = document.getElementById("cb"+CLASSES.indexOf(SPECIAL[j]));
                if (sb) sb.classList.remove("sel");
            }
        }
    }
    var pos = sel.indexOf(cls);
    var btn = document.getElementById("cb"+i);
    if (pos >= 0) { sel.splice(pos,1); btn.classList.remove("sel"); }
    else          { sel.push(cls);     btn.classList.add("sel"); }
}

// ── Load patch ────────────────────────────────────────────────
function loadPatch(i) {
    curIdx = i;
    var p    = patches[i];
    var pred = getPrediction(p);
    var val  = getValidated(p);

    document.getElementById("glom-img").src = "/patch/"+encodeURIComponent(p);
    document.getElementById("pname").textContent  = p;
    document.getElementById("pidx").textContent   = (i+1)+" of "+patches.length;
    var m = META[p];
    document.getElementById("pslide").textContent = m ? "Slide: "+m.slide : "";

    // Status badge
    var badge   = document.getElementById("pred-badge");
    var status  = getStatus(p);
    badge.className = "pred-badge" + (status==="confirmed"?" confirmed":status==="corrected"?" corrected":"");
    badge.textContent = status==="confirmed" ? "✓ Confirmed" :
                        status==="corrected" ? "✏ Corrected" :
                        "Model: "+(pred.length ? pred.join(", ") : "Unclassified");

    // Load selection: validated if exists, else model prediction
    sel = val ? val.slice() : pred.slice();
    renderCls(pred);
    updateProgress();
    document.getElementById("saved-msg").textContent = "";

    // Update strip
    var imgs = document.querySelectorAll("#tstrip img");
    for (var j = 0; j < imgs.length; j++) {
        updateThumbColor(j, imgs[j]);
    }
    if (imgs[i]) imgs[i].scrollIntoView({block:"nearest",inline:"center"});

    var img = document.getElementById("glom-img");
    img.onload = function(){ drawOverlay(); };
    if (img.complete) drawOverlay();
}

// ── Save ──────────────────────────────────────────────────────
function doSave(callback) {
    if (!user) { if (callback) callback(); return; }
    var p    = patches[curIdx];
    var lbls = sel.slice();
    var pred = getPrediction(p).join("|");

    fetch("/save", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({patch:p, labels:lbls, user:user, prediction:pred})
    })
    .then(function(r){ if(!r.ok) throw new Error(r.status); return r.json(); })
    .then(function(data){
        var status = data.status_label || "validated";
        VALIDATED[p] = {labels:lbls.join("|"), user:user, status:status};
        var msg = document.getElementById("saved-msg");
        msg.textContent = status==="confirmed" ? "✓ Confirmed" : "✏ Corrected";
        msg.className   = "saved-msg ok";
        var imgs = document.querySelectorAll("#tstrip img");
        var savedI = patches.indexOf(p);
        if (savedI>=0) updateThumbColor(savedI, imgs[savedI]);
        updateProgress();
        if (callback) callback();
    })
    .catch(function(e){
        var msg = document.getElementById("saved-msg");
        msg.textContent = "❌ Error: "+e.message;
        msg.className   = "saved-msg err";
    });
}

function doSaveNav(dir) {
    doSave(function(){
        var ni = curIdx+dir;
        if (ni<0) ni=patches.length-1;
        if (ni>=patches.length) ni=0;
        loadPatch(ni);
    });
}
function doSaveGoTo(i) { doSave(function(){ loadPatch(i); }); }

// ── Progress ──────────────────────────────────────────────────
function updateProgress() {
    var done = patches.filter(function(p){ return hasLabel(p); }).length;
    var pct  = Math.round(done/patches.length*100);
    var conf = patches.filter(function(p){ return getStatus(p)==="confirmed"; }).length;
    var corr = patches.filter(function(p){ return getStatus(p)==="corrected"; }).length;
    document.getElementById("stats").textContent =
        done+" / "+patches.length+" validated — "+conf+" confirmed, "+corr+" corrected ("+pct+"%)";
    document.getElementById("pfill").style.width = pct+"%";
}

// ── Thumb color ───────────────────────────────────────────────
function updateThumbColor(i, img) {
    if (!img) return;
    var p = patches[i];
    if (i===curIdx) {
        img.style.borderColor="#e94560"; img.style.opacity="1";
    } else {
        var st = getStatus(p);
        if (st==="confirmed")    { img.style.borderColor="#4caf50"; img.style.opacity=".85"; }
        else if (st==="corrected"){ img.style.borderColor="#ff9800"; img.style.opacity=".85"; }
        else if (PREDICTIONS[p]) { img.style.borderColor="#2196F3"; img.style.opacity=".7";  }
        else                     { img.style.borderColor="transparent"; img.style.opacity=".45"; }
    }
}

// ── Overlay ────────────────────────────────────────────────────
function drawOverlay() {
    var img=document.getElementById("glom-img");
    var cv=document.getElementById("overlay-canvas");
    cv.width=img.naturalWidth; cv.height=img.naturalHeight;
    cv.style.width=img.offsetWidth+"px"; cv.style.height=img.offsetHeight+"px";
    var m=META[patches[curIdx]];
    if (!m||!m.polygon||m.polygon.length<3) return;
    var ctx=cv.getContext("2d");
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.beginPath();
    for (var i=0;i<m.polygon.length;i++){
        var pt=m.polygon[i];
        if(i===0) ctx.moveTo(pt[0],pt[1]); else ctx.lineTo(pt[0],pt[1]);
    }
    ctx.closePath();
    ctx.strokeStyle="rgba(255,220,0,.65)"; ctx.lineWidth=2;
    ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
}

function openZoom() {
    var img=document.getElementById("glom-img");
    var cv=document.getElementById("zoom-canvas");
    var sc=Math.min(window.innerWidth*.9/img.naturalWidth, window.innerHeight*.9/img.naturalHeight);
    cv.width=img.naturalWidth*sc; cv.height=img.naturalHeight*sc;
    var ctx=cv.getContext("2d");
    ctx.drawImage(img,0,0,cv.width,cv.height);
    var m=META[patches[curIdx]];
    if (m&&m.polygon) {
        ctx.beginPath();
        for(var i=0;i<m.polygon.length;i++){
            var pt=m.polygon[i];
            if(i===0) ctx.moveTo(pt[0]*sc,pt[1]*sc); else ctx.lineTo(pt[0]*sc,pt[1]*sc);
        }
        ctx.closePath();
        ctx.strokeStyle="rgba(255,220,0,.65)"; ctx.lineWidth=2;
        ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
    }
    document.getElementById("zoom-overlay").classList.add("vis");
}
function closeZoom(){ document.getElementById("zoom-overlay").classList.remove("vis"); }

// ── Keyboard ──────────────────────────────────────────────────
document.addEventListener("keydown", function(e){
    if (document.getElementById("login-screen").style.display!=="none") return;
    if (e.target.tagName==="INPUT") return;
    if (e.key==="ArrowRight") doSaveNav(1);
    else if (e.key==="ArrowLeft") doSaveNav(-1);
    else if (e.key===" "){
        e.preventDefault();
        if(document.getElementById("zoom-overlay").classList.contains("vis")) closeZoom(); else openZoom();
    }
    else if (e.key==="Escape") closeZoom();
    else if (e.key>="1"&&e.key<="9") toggleCls(CLASSES[parseInt(e.key)-1], parseInt(e.key)-1);
    else if (e.key==="0") toggleCls(CLASSES[9],9);
});

// ── Thumbnail strip ───────────────────────────────────────────
var strip = document.getElementById("tstrip");
for (var ti=0; ti<patches.length; ti++) {
    var timg=document.createElement("img");
    timg.src="/patch/"+encodeURIComponent(patches[ti]);
    timg.loading="lazy";
    (function(i2){ timg.addEventListener("click",function(){ doSaveGoTo(i2); }); })(ti);
    strip.appendChild(timg);
}
// Init colors
var allImgs = document.querySelectorAll("#tstrip img");
for (var k=0;k<allImgs.length;k++) updateThumbColor(k, allImgs[k]);
</script>
</body></html>"""

@app.route('/')
def index():
    return render_template_string(HTML,
        classes=CLASSES,
        patches=get_patches(),
        predictions=load_predictions(),
        validated=load_validated(),
        metadata=load_meta())

@app.route('/patch/<path:filename>')
def serve_patch(filename):
    path = os.path.join(PATCHES_DIR, filename)
    if not os.path.exists(path): return "Not found", 404
    return send_file(path)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json(force=True, silent=True)
    if not data or 'patch' not in data:
        return jsonify({'status':'error'}), 400
    status = save_validated(
        data['patch'],
        data.get('labels', []),
        data.get('user', '?'),
        data.get('prediction', '')
    )
    return jsonify({'status':'ok', 'status_label': status})

if __name__ == '__main__':
    patches   = get_patches()
    validated = load_validated()
    preds     = load_predictions()
    print(f"✓ {len(patches)} patches")
    print(f"✓ {len(preds)} model predictions loaded")
    print(f"✓ {len(validated)} already validated")
    import socket
    print(f"→ http://localhost:5000")
    print(f"→ http://{socket.gethostbyname(socket.gethostname())}:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
