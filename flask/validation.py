from flask import Flask, render_template_string, request, jsonify, send_file
import pandas as pd
import os, json, random, datetime
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

PROB_COLS = ["prob_Normal","prob_Adhesion","prob_Thickening_GBM",
             "prob_Fibrinoid_necrosis","prob_Hypercellularity",
             "prob_Fibrosis","prob_Crescent","prob_Sclerosis"]
CLASSES_SHORT = ["Normal","Adhesion","Thick.GBM","Fibrinoid",
                 "Hypercell.","Fibrosis","Crescent","Sclerosis"]
COLORS = ["#52c41a","#fa8c16","#1890ff","#ff4d4f",
          "#722ed1","#fa541c","#eb2f96","#8c8c8c"]
THRESHOLD = 0.5

Path(VALIDATED_FILE).parent.mkdir(parents=True, exist_ok=True)

# ── Data helpers ───────────────────────────────────────────────
def load_predictions():
    if not os.path.exists(PREDICTIONS_FILE): return {}
    df = pd.read_csv(PREDICTIONS_FILE)
    result = {}
    for _, row in df.iterrows():
        lbl = str(row['classes']) if pd.notna(row['classes']) else ''
        if lbl and lbl not in ('nan','Unclassified'):
            result[row['patch']] = lbl
    return result

def load_validated():
    if not os.path.exists(VALIDATED_FILE): return {}
    df = pd.read_csv(VALIDATED_FILE)
    result = {}
    for _, row in df.iterrows():
        lbl = str(row['labels']) if pd.notna(row['labels']) else ''
        if lbl and lbl != 'nan':
            result[row['patch']] = {
                'labels': lbl,
                'user':   str(row.get('user','')),
                'status': str(row.get('status','validated'))
            }
    return result

def save_validated(patch, labels, user, original_pred):
    data = load_validated()
    pred_set = set(original_pred.split('|')) if original_pred else set()
    new_set  = set(labels)
    status   = 'confirmed' if pred_set == new_set else 'corrected'
    data[patch] = {'labels':'|'.join(labels),'user':user,'status':status}
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

def compute_results():
    pred_df = pd.read_csv(PREDICTIONS_FILE) if os.path.exists(PREDICTIONS_FILE) else pd.DataFrame()
    val_df  = pd.read_csv(VALIDATED_FILE)   if os.path.exists(VALIDATED_FILE)   else pd.DataFrame()

    slides = {}
    if not pred_df.empty:
        for slide, sdf in pred_df.groupby("slide"):
            n = len(sdf)
            model_pct = [round(100*(sdf[c]>=THRESHOLD).sum()/n,1) for c in PROB_COLS]
            slides[slide] = {
                "n_total":   n,
                "model_pct": model_pct,
                "human_pct": [0]*8,
                "n_val":0,"n_conf":0,"n_corr":0,
                "type": "kidney" if "Kidney" in slide else "lym"
            }

    n_confirmed = 0; n_corrected = 0
    if not val_df.empty and "status" in val_df.columns:
        n_confirmed = int((val_df["status"]=="confirmed").sum())
        n_corrected = int((val_df["status"]=="corrected").sum())
        if "labels" in val_df.columns and not pred_df.empty:
            val_df["slide"] = val_df["patch"].apply(
                lambda x: "_".join(str(x).split("_")[:-1]) if "LysM" in str(x)
                          else str(x).rsplit("_",1)[0])
            for slide, sdf in val_df.groupby("slide"):
                if slide not in slides: continue
                n = len(sdf)
                slides[slide]["n_val"]  = n
                slides[slide]["n_conf"] = int((sdf["status"]=="confirmed").sum())
                slides[slide]["n_corr"] = int((sdf["status"]=="corrected").sum())
                human_pct = []
                for cls in ["Normal","Adhesion","Thickening GBM","Fibrinoid necrosis",
                            "Hypercellularity","Fibrosis","Crescent","Sclerosis"]:
                    n_pos = sdf["labels"].apply(lambda l: cls in str(l).split("|")).sum()
                    human_pct.append(round(100*n_pos/n,1) if n>0 else 0)
                slides[slide]["human_pct"] = human_pct
    return slides, n_confirmed, n_corrected

# ── HTML templates ─────────────────────────────────────────────
VALIDATION_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Glomerulus Validator</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;800&display=swap');
:root{--bg:#0b0c10;--surface:#12141a;--border:#1e2130;--accent:#7ee8fa;--green:#52c41a;--orange:#ff9800;--text:#e8eaf0;--muted:#4a5068}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Syne',sans-serif;background:var(--bg);color:var(--text)}
#login-screen{position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;z-index:1000}
.login-box{background:var(--surface);border:2px solid var(--border);border-radius:16px;padding:40px;text-align:center;width:400px}
.login-box h1{color:var(--accent);margin-bottom:8px;font-size:1.4em}
.login-box p{color:var(--muted);margin-bottom:20px;font-size:.85em;line-height:1.6}
.login-box input{width:100%;padding:12px 16px;border-radius:8px;border:2px solid var(--border);background:var(--bg);color:var(--text);font-size:1em;margin-bottom:12px;outline:none}
.login-box input:focus{border-color:var(--accent)}
.login-box button{width:100%;padding:12px;background:var(--accent);color:var(--bg);border:none;border-radius:8px;font-size:1em;font-weight:700;cursor:pointer}
header{background:var(--surface);padding:10px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border)}
header h1{font-size:1.05em;color:var(--accent)}
.hright{display:flex;align-items:center;gap:16px}
.ubadge{font-size:.78em;color:var(--green)}
.stats{font-size:.78em;color:var(--muted)}
.pbar{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:3px;width:180px}
.pfill{height:100%;background:var(--accent);transition:width .3s}
.results-btn{font-family:'DM Mono',monospace;font-size:.72em;color:var(--accent);
  text-decoration:none;border:1px solid var(--accent);padding:4px 14px;
  border-radius:12px;transition:all .2s}
.results-btn:hover{background:var(--accent);color:var(--bg)}
.main{display:flex;height:calc(100vh - 130px)}
.ipanel{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px;background:var(--surface)}
.cbox{position:relative;cursor:zoom-in;border-radius:8px;border:3px solid var(--border);overflow:hidden}
#glom-img{display:block;max-width:100%;max-height:52vh;image-rendering:pixelated}
#overlay-canvas{position:absolute;top:0;left:0;pointer-events:none}
.pinfo{margin-top:10px;text-align:center}
.pname{font-size:.78em;color:#888}
.pidx{font-size:.72em;color:var(--muted);margin-top:2px}
.pslide{font-size:.7em;color:var(--muted);margin-top:1px}
.pred-badge{margin-top:8px;padding:4px 12px;border-radius:12px;font-family:'DM Mono',monospace;
  font-size:.72em;display:inline-block;border:1px solid var(--muted);color:var(--muted)}
.pred-badge.confirmed{border-color:var(--green);color:var(--green)}
.pred-badge.corrected{border-color:var(--orange);color:var(--orange)}
.navbtns{display:flex;gap:12px;margin-top:14px}
.nbtn{padding:10px 22px;border:none;background:var(--accent);color:var(--bg);border-radius:8px;cursor:pointer;font-size:.9em;font-weight:700}
.nbtn:hover{opacity:.85}
.hints{margin-top:8px;font-size:.68em;color:var(--muted);text-align:center}
.hints kbd{background:var(--border);color:var(--muted);border-radius:3px;padding:1px 5px}
.cpanel{width:310px;background:var(--surface);border-left:2px solid var(--border);padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:7px}
.cpanel h2{font-size:.88em;color:#888;margin-bottom:2px}
.cpanel .sub{font-family:'DM Mono',monospace;font-size:.68em;color:var(--muted);margin-bottom:6px}
.sep{height:1px;background:var(--border);margin:4px 0}
.cbtn{display:flex;align-items:center;gap:10px;padding:9px 14px;border-radius:8px;border:2px solid var(--border);background:var(--bg);color:var(--text);cursor:pointer;font-size:.87em;transition:all .15s;user-select:none}
.cbtn:hover{border-color:var(--accent);background:#1a2a3a}
.cbtn.sel{border-color:var(--accent);background:#1a3040;color:#fff}
.cbtn.sp{border-color:#2a2a4e}
.cbtn.sp:hover{border-color:var(--orange);background:#2a1a0e}
.cbtn.sp.sel{border-color:var(--orange);background:#3d2a0e}
.cbtn.sp.sel .dot{background:var(--orange);border-color:var(--orange)}
.cbtn.sp.sel .kn{color:var(--orange)}
.cbtn.predicted:not(.sel){border-color:#1a4060;background:rgba(33,150,243,.06)}
.kn{font-size:.73em;color:var(--muted);background:var(--border);border-radius:3px;padding:1px 5px;min-width:20px;text-align:center}
.cbtn.sel .kn{color:var(--accent)}
.dot{width:13px;height:13px;border-radius:50%;border:2px solid var(--muted);flex-shrink:0}
.cbtn.sel .dot{background:var(--accent);border-color:var(--accent)}
.saved-msg{text-align:center;font-family:'DM Mono',monospace;font-size:.78em;height:18px;margin-top:4px}
.saved-msg.ok{color:var(--green)}
.saved-msg.err{color:var(--accent)}
#zoom-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:500;align-items:center;justify-content:center;cursor:zoom-out}
#zoom-overlay.vis{display:flex}
#zoom-canvas{max-width:90vw;max-height:90vh;border-radius:8px}
#tstrip{position:fixed;bottom:0;left:0;right:0;height:68px;background:var(--border);display:flex;overflow-x:auto;gap:3px;padding:4px;align-items:center;z-index:1}
#tstrip img{display:block;height:58px;width:58px;object-fit:cover;border-radius:4px;border:2px solid transparent;opacity:.45;cursor:pointer;flex-shrink:0}
.legend-strip{position:fixed;bottom:70px;right:8px;font-family:'DM Mono',monospace;font-size:.62em;color:var(--muted);text-align:right;line-height:2}
.ld{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:3px}
</style></head><body>

<div id="login-screen">
  <div class="login-box">
    <h1>&#128302; Glomerulus Validator</h1>
    <p>Model predictions are pre-checked.<br>Confirm or correct each glomerulus.<br>
    <span style="color:var(--accent)">&#128309;</span> = model predicted this class</p>
    <input id="uname" type="text" placeholder="Your name..." maxlength="30">
    <button id="start-btn">Start validation &#8594;</button>
  </div>
</div>

<header>
  <h1>&#128302; Glomerulus Validator</h1>
  <div class="hright">
    <div>
      <div class="ubadge" id="ubadge"></div>
      <div class="stats" id="stats">Loading...</div>
      <div class="pbar"><div class="pfill" id="pfill" style="width:0%"></div></div>
    </div>
    <a href="/results" target="_blank" class="results-btn">&#128200; Live results</a>
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
    <div class="sub">&#128309; = model prediction</div>
    <div id="cbts"></div>
    <div class="saved-msg" id="saved-msg"></div>
  </div>
</div>

<div id="zoom-overlay"><canvas id="zoom-canvas"></canvas></div>
<div id="tstrip"></div>
<div class="legend-strip">
  <span class="ld" style="background:#333"></span>unseen &nbsp;
  <span class="ld" style="background:#2196F3"></span>predicted &nbsp;
  <span class="ld" style="background:#52c41a"></span>confirmed &nbsp;
  <span class="ld" style="background:#ff9800"></span>corrected
</div>

<script>
var CLASSES={{ classes|tojson }};
var SPECIAL=["Double glomerulus","Not a glom"];
var patches={{ patches|tojson }};
var META={{ metadata|tojson }};
var PREDICTIONS={{ predictions|tojson }};
var VALIDATED={{ validated|tojson }};
var curIdx=0,sel=[],user="";

function getPred(p){var v=PREDICTIONS[p]||"";return v&&v!=="Unclassified"?v.split("|").filter(Boolean):[];}
function getVal(p){var v=VALIDATED[p];return v?v.labels.split("|").filter(Boolean):null;}
function getStatus(p){var v=VALIDATED[p];return v?v.status||"validated":"unseen";}
function hasLabel(p){return !!VALIDATED[p];}

document.getElementById("start-btn").addEventListener("click",function(){
  var n=document.getElementById("uname").value.trim();
  if(!n){alert("Please enter your name");return;}
  user=n;
  document.getElementById("login-screen").style.display="none";
  document.getElementById("ubadge").textContent="👤 "+n;
  var si=0;for(var i=0;i<patches.length;i++){if(!hasLabel(patches[i])){si=i;break;}}
  loadPatch(si);
});
document.getElementById("uname").addEventListener("keydown",function(e){if(e.key==="Enter")document.getElementById("start-btn").click();});
document.getElementById("prev-btn").addEventListener("click",function(){doSaveNav(-1);});
document.getElementById("next-btn").addEventListener("click",function(){doSaveNav(1);});
document.getElementById("cbox").addEventListener("click",openZoom);
document.getElementById("zoom-overlay").addEventListener("click",closeZoom);

function renderCls(pred){
  var c=document.getElementById("cbts");c.innerHTML="";
  for(var i=0;i<CLASSES.length;i++){
    var cls=CLASSES[i],isSp=SPECIAL.indexOf(cls)>=0,isSel=sel.indexOf(cls)>=0,isPred=pred.indexOf(cls)>=0;
    if(cls==="Double glomerulus"){var s=document.createElement("div");s.className="sep";c.appendChild(s);}
    var btn=document.createElement("div");
    btn.className="cbtn"+(isSp?" sp":"")+(isSel?" sel":"")+(isPred&&!isSel?" predicted":"");
    btn.id="cb"+i;
    var k=i<9?(i+1):0;
    var dot=isPred?'<span style="color:#2196F3;font-size:.65em;margin-left:2px">●</span>':'';
    btn.innerHTML='<div class="dot"></div><span style="flex:1">'+cls+dot+'</span><span class="kn">'+k+'</span>';
    (function(c2,i2){btn.addEventListener("click",function(){toggleCls(c2,i2);});})(cls,i);
    c.appendChild(btn);
  }
}

function toggleCls(cls,i){
  var isSp=SPECIAL.indexOf(cls)>=0,btn=document.getElementById("cb"+i);
  if(isSp&&sel.indexOf(cls)<0){sel=[];document.querySelectorAll(".cbtn.sel").forEach(function(b){b.classList.remove("sel");});}
  if(!isSp){SPECIAL.forEach(function(sc){var idx=sel.indexOf(sc);if(idx>=0){sel.splice(idx,1);var sb=document.getElementById("cb"+CLASSES.indexOf(sc));if(sb)sb.classList.remove("sel");}});}
  var pos=sel.indexOf(cls);
  if(pos>=0){sel.splice(pos,1);btn.classList.remove("sel");}else{sel.push(cls);btn.classList.add("sel");}
}

function loadPatch(i){
  curIdx=i;var p=patches[i],pred=getPred(p),val=getVal(p),status=getStatus(p);
  document.getElementById("glom-img").src="/patch/"+encodeURIComponent(p);
  document.getElementById("pname").textContent=p;
  document.getElementById("pidx").textContent=(i+1)+" of "+patches.length;
  var m=META[p];document.getElementById("pslide").textContent=m?"Slide: "+m.slide:"";
  var badge=document.getElementById("pred-badge");
  badge.className="pred-badge"+(status==="confirmed"?" confirmed":status==="corrected"?" corrected":"");
  badge.textContent=status==="confirmed"?"✓ Confirmed":status==="corrected"?"✏ Corrected":"Model: "+(pred.length?pred.join(", "):"Unclassified");
  sel=val?val.slice():pred.slice();
  renderCls(pred);updateProgress();document.getElementById("saved-msg").textContent="";
  var imgs=document.querySelectorAll("#tstrip img");
  for(var j=0;j<imgs.length;j++)updateThumbColor(j,imgs[j]);
  if(imgs[i])imgs[i].scrollIntoView({block:"nearest",inline:"center"});
  var img=document.getElementById("glom-img");
  img.onload=function(){drawOverlay();};if(img.complete)drawOverlay();
}

function doSave(callback){
  if(!user){if(callback)callback();return;}
  var p=patches[curIdx],lbls=sel.slice(),pred=getPred(p).join("|");
  fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({patch:p,labels:lbls,user:user,prediction:pred})})
  .then(function(r){if(!r.ok)throw new Error(r.status);return r.json();})
  .then(function(data){
    var st=data.status_label||"validated";
    VALIDATED[p]={labels:lbls.join("|"),user:user,status:st};
    var msg=document.getElementById("saved-msg");
    msg.textContent=st==="confirmed"?"✓ Confirmed":"✏ Corrected";
    msg.className="saved-msg ok";
    var imgs=document.querySelectorAll("#tstrip img");
    var si=patches.indexOf(p);if(si>=0)updateThumbColor(si,imgs[si]);
    updateProgress();if(callback)callback();
  })
  .catch(function(e){var msg=document.getElementById("saved-msg");msg.textContent="❌ "+e.message;msg.className="saved-msg err";});
}

function doSaveNav(dir){doSave(function(){var ni=curIdx+dir;if(ni<0)ni=patches.length-1;if(ni>=patches.length)ni=0;loadPatch(ni);});}
function doSaveGoTo(i){doSave(function(){loadPatch(i);});}

function updateProgress(){
  var done=patches.filter(function(p){return hasLabel(p);}).length;
  var pct=Math.round(done/patches.length*100);
  document.getElementById("stats").textContent=done+" / "+patches.length+" validated ("+pct+"%)";
  document.getElementById("pfill").style.width=pct+"%";
}

function updateThumbColor(i,img){
  if(!img)return;var p=patches[i];
  if(i===curIdx){img.style.borderColor="#e94560";img.style.opacity="1";}
  else{var st=getStatus(p);
    if(st==="confirmed"){img.style.borderColor="#52c41a";img.style.opacity=".85";}
    else if(st==="corrected"){img.style.borderColor="#ff9800";img.style.opacity=".85";}
    else if(PREDICTIONS[p]){img.style.borderColor="#2196F3";img.style.opacity=".65";}
    else{img.style.borderColor="transparent";img.style.opacity=".4";}
  }
}

function drawOverlay(){
  var img=document.getElementById("glom-img"),cv=document.getElementById("overlay-canvas");
  cv.width=img.naturalWidth;cv.height=img.naturalHeight;
  cv.style.width=img.offsetWidth+"px";cv.style.height=img.offsetHeight+"px";
  var m=META[patches[curIdx]];if(!m||!m.polygon||m.polygon.length<3)return;
  var ctx=cv.getContext("2d");ctx.clearRect(0,0,cv.width,cv.height);ctx.beginPath();
  for(var i=0;i<m.polygon.length;i++){var pt=m.polygon[i];i===0?ctx.moveTo(pt[0],pt[1]):ctx.lineTo(pt[0],pt[1]);}
  ctx.closePath();ctx.strokeStyle="rgba(255,220,0,.65)";ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);
}

function openZoom(){
  var img=document.getElementById("glom-img"),cv=document.getElementById("zoom-canvas");
  var sc=Math.min(window.innerWidth*.9/img.naturalWidth,window.innerHeight*.9/img.naturalHeight);
  cv.width=img.naturalWidth*sc;cv.height=img.naturalHeight*sc;
  var ctx=cv.getContext("2d");ctx.drawImage(img,0,0,cv.width,cv.height);
  var m=META[patches[curIdx]];
  if(m&&m.polygon){ctx.beginPath();for(var i=0;i<m.polygon.length;i++){var pt=m.polygon[i];i===0?ctx.moveTo(pt[0]*sc,pt[1]*sc):ctx.lineTo(pt[0]*sc,pt[1]*sc);}
    ctx.closePath();ctx.strokeStyle="rgba(255,220,0,.65)";ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);}
  document.getElementById("zoom-overlay").classList.add("vis");
}
function closeZoom(){document.getElementById("zoom-overlay").classList.remove("vis");}

document.addEventListener("keydown",function(e){
  if(document.getElementById("login-screen").style.display!=="none")return;
  if(e.target.tagName==="INPUT")return;
  if(e.key==="ArrowRight")doSaveNav(1);
  else if(e.key==="ArrowLeft")doSaveNav(-1);
  else if(e.key===" "){e.preventDefault();document.getElementById("zoom-overlay").classList.contains("vis")?closeZoom():openZoom();}
  else if(e.key==="Escape")closeZoom();
  else if(e.key>="1"&&e.key<="9")toggleCls(CLASSES[parseInt(e.key)-1],parseInt(e.key)-1);
  else if(e.key==="0")toggleCls(CLASSES[9],9);
});

var strip=document.getElementById("tstrip");
for(var ti=0;ti<patches.length;ti++){
  var timg=document.createElement("img");
  timg.src="/patch/"+encodeURIComponent(patches[ti]);timg.loading="lazy";
  (function(i2){timg.addEventListener("click",function(){doSaveGoTo(i2);});})(ti);
  strip.appendChild(timg);
}
var allImgs=document.querySelectorAll("#tstrip img");
for(var k=0;k<allImgs.length;k++)updateThumbColor(k,allImgs[k]);
</script></body></html>"""

RESULTS_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Live Results</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;800&display=swap');
:root{--bg:#0b0c10;--surface:#12141a;--border:#1e2130;--accent:#7ee8fa;--green:#52c41a;--orange:#ff9800;--text:#e8eaf0;--muted:#4a5068}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(126,232,250,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(126,232,250,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none}
header{padding:24px 40px 18px;border-bottom:1px solid var(--border);display:flex;align-items:flex-end;justify-content:space-between}
.title h1{font-size:1.9em;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.title p{font-family:'DM Mono',monospace;font-size:.7em;color:var(--muted);margin-top:4px;letter-spacing:.05em}
.refresh{font-family:'DM Mono',monospace;font-size:.65em;color:var(--muted)}
.gstats{display:flex;gap:0;border-bottom:1px solid var(--border)}
.gstat{flex:1;padding:18px 28px;border-right:1px solid var(--border)}
.gstat:last-child{border-right:none}
.gstat .val{font-size:2em;font-weight:800;line-height:1}
.gstat .lbl{font-family:'DM Mono',monospace;font-size:.65em;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:4px}
.gstat .sub{font-family:'DM Mono',monospace;font-size:.68em;color:var(--muted);margin-top:6px}
.ptrack{height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;margin-top:8px}
.pfill{height:100%;border-radius:3px}
.legend{display:flex;gap:20px;padding:14px 40px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.li{display:flex;align-items:center;gap:6px}
.ll{width:22px;height:3px;border-radius:2px}
.lt{font-family:'DM Mono',monospace;font-size:.67em;color:var(--muted)}
.main{padding:28px 40px}
.sec{font-family:'DM Mono',monospace;font-size:.68em;color:var(--muted);text-transform:uppercase;letter-spacing:.15em;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;position:relative;overflow:hidden}
.card.kidney{border-color:#1e3a1e}
.card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.card.lym::after{background:linear-gradient(90deg,var(--accent),transparent)}
.card.kidney::after{background:linear-gradient(90deg,var(--green),transparent)}
.ctop{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px}
.cname{font-weight:800;font-size:.95em}
.cbadge{font-family:'DM Mono',monospace;font-size:.58em;padding:2px 8px;border-radius:8px}
.cbadge.lym{background:rgba(126,232,250,.1);color:var(--accent)}
.cbadge.kidney{background:rgba(82,196,26,.1);color:var(--green)}
.prow{display:flex;justify-content:space-between;margin-bottom:5px}
.plbl{font-family:'DM Mono',monospace;font-size:.67em;color:var(--muted)}
.pcount{font-family:'DM Mono',monospace;font-size:.67em;color:var(--text)}
.chips{display:flex;gap:6px;margin:8px 0 12px}
.chip{font-family:'DM Mono',monospace;font-size:.6em;padding:2px 8px;border-radius:6px}
.conf{background:rgba(82,196,26,.12);color:var(--green)}
.corr{background:rgba(255,152,0,.12);color:var(--orange)}
.pend{background:rgba(74,80,104,.2);color:var(--muted)}
.radars{display:flex;gap:12px;justify-content:center;align-items:center;margin:4px 0}
.rb{text-align:center}
.rb canvas{display:block}
.rl{font-family:'DM Mono',monospace;font-size:.58em;margin-top:3px}
.rl.m{color:rgba(126,232,250,.55)}
.rl.h{color:rgba(82,196,26,.55)}
.rvs{font-size:1.1em;color:var(--muted);font-weight:800}
.bars{margin-top:12px;display:flex;flex-direction:column;gap:4px}
.br{display:flex;align-items:center;gap:5px}
.bl{font-family:'DM Mono',monospace;font-size:.57em;color:var(--muted);width:78px;flex-shrink:0;text-align:right}
.bt{flex:1;height:8px;border-radius:4px;background:rgba(255,255,255,.05);position:relative;overflow:hidden}
.bm{height:100%;border-radius:4px;opacity:.4;position:absolute;top:0;left:0}
.bh{height:4px;border-radius:2px;position:absolute;bottom:0;left:0}
.bv{font-family:'DM Mono',monospace;font-size:.57em;color:var(--muted);width:30px;text-align:right}
.nodata{text-align:center;padding:18px;font-family:'DM Mono',monospace;font-size:.72em;color:var(--muted)}
</style></head><body>
<header>
  <div class="title">
    <h1>Live Validation Results</h1>
    <p>MODEL PREDICTIONS vs HUMAN VALIDATION · AUTO-REFRESH 30s</p>
  </div>
  <div class="refresh">Updated: __TIMESTAMP__</div>
</header>
<div class="gstats">
  <div class="gstat"><div class="val" style="color:var(--accent)">__TOTAL_VAL__</div><div class="lbl">Patches validated</div><div class="sub">of __TOTAL_P__ total</div><div class="ptrack"><div class="pfill" style="width:__PCT_DONE__%;background:var(--accent)"></div></div></div>
  <div class="gstat"><div class="val" style="color:var(--green)">__PCT_CONF__%</div><div class="lbl">Confirmed</div><div class="sub">__N_CONF__ — model correct</div></div>
  <div class="gstat"><div class="val" style="color:var(--orange)">__PCT_CORR__%</div><div class="lbl">Corrected</div><div class="sub">__N_CORR__ — human override</div></div>
  <div class="gstat"><div class="val">__N_STARTED__/__N_SLIDES__</div><div class="lbl">Slides started</div><div class="sub">__N_DONE__ fully validated</div></div>
</div>
<div class="legend">
  <div class="li"><div class="ll" style="background:rgba(126,232,250,.5)"></div><div class="lt">Model prediction (top, all patches)</div></div>
  <div class="li"><div class="ll" style="background:var(--green)"></div><div class="lt">Human validation (bottom, validated patches)</div></div>
</div>
<div class="main">
  <div class="sec">Per-slide profiles</div>
  <div class="grid">__CARDS__</div>
</div>
<script>
const COLORS=["#52c41a","#fa8c16","#1890ff","#ff4d4f","#722ed1","#fa541c","#eb2f96","#8c8c8c"];
const SHORT=["Normal","Adhesion","Thick.GBM","Fibrinoid","Hypercell.","Fibrosis","Crescent","Sclerosis"];
function drawR(id,vals,col,fa){
  var cv=document.getElementById(id);if(!cv)return;
  var ctx=cv.getContext("2d"),sz=110,cx=sz/2,cy=sz/2,R=42,n=vals.length;
  cv.width=sz;cv.height=sz;
  [25,50,75,100].forEach(function(r){ctx.beginPath();for(var i=0;i<n;i++){var a=(i/n)*Math.PI*2-Math.PI/2,x=cx+Math.cos(a)*(R*r/100),y=cy+Math.sin(a)*(R*r/100);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.closePath();ctx.strokeStyle="rgba(255,255,255,.05)";ctx.lineWidth=1;ctx.stroke();});
  for(var i=0;i<n;i++){var a=(i/n)*Math.PI*2-Math.PI/2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(a)*R,cy+Math.sin(a)*R);ctx.strokeStyle="rgba(255,255,255,.06)";ctx.lineWidth=1;ctx.stroke();}
  if(vals.some(function(v){return v>0;})){
    ctx.beginPath();vals.forEach(function(v,i){var a=(i/n)*Math.PI*2-Math.PI/2,r=R*(v/100);i===0?ctx.moveTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r):ctx.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r);});
    ctx.closePath();ctx.fillStyle=col.replace("rgb(","rgba(").replace(")",","+fa+")");ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
    vals.forEach(function(v,i){if(v<=0)return;var a=(i/n)*Math.PI*2-Math.PI/2;ctx.beginPath();ctx.arc(cx+Math.cos(a)*R*(v/100),cy+Math.sin(a)*R*(v/100),2,0,Math.PI*2);ctx.fillStyle=COLORS[i];ctx.fill();});
  }else{ctx.font="8px DM Mono";ctx.fillStyle="rgba(255,255,255,.15)";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("no data",cx,cy);}
}
document.querySelectorAll("[data-mv]").forEach(function(el){
  var sid=el.dataset.sid,mv=JSON.parse(el.dataset.mv),hv=JSON.parse(el.dataset.hv);
  drawR("rm"+sid,mv,"rgb(126,232,250)",.15);
  drawR("rh"+sid,hv,"rgb(82,196,26)",.2);
});
</script></body></html>"""

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(VALIDATION_HTML,
        classes=CLASSES, patches=get_patches(),
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
    status = save_validated(data['patch'], data.get('labels',[]),
                            data.get('user','?'), data.get('prediction',''))
    return jsonify({'status':'ok','status_label':status})

@app.route('/results')
def results():
    slides, n_confirmed, n_corrected = compute_results()
    total_patches   = sum(s["n_total"] for s in slides.values())
    total_validated = n_confirmed + n_corrected
    pct_done        = round(100*total_validated/total_patches,1) if total_patches else 0
    pct_conf        = round(100*n_confirmed/total_validated,1)   if total_validated else 0
    pct_corr        = round(100*n_corrected/total_validated,1)   if total_validated else 0
    n_slides        = len(slides)
    n_started       = sum(1 for s in slides.values() if s["n_val"]>0)
    n_done          = sum(1 for s in slides.values() if s["n_val"]>=s["n_total"])
    ts              = datetime.datetime.now().strftime("%H:%M:%S")

    cards = ""
    CSHORT = ["Normal","Adhesion","Thick.GBM","Fibrinoid","Hypercell.","Fibrosis","Crescent","Sclerosis"]
    CCOLORS= ["#52c41a","#fa8c16","#1890ff","#ff4d4f","#722ed1","#fa541c","#eb2f96","#8c8c8c"]

    for slide, s in sorted(slides.items(), key=lambda x:(0 if x[1]["type"]=="lym" else 1, x[0])):
        sid   = slide.replace(" ","_").replace(".","_")
        is_k  = s["type"]=="kidney"
        badge = "kidney" if is_k else "lym"
        lbl   = "WT" if is_k else "LysM"
        nv,nt = s["n_val"],s["n_total"]
        pct   = round(100*nv/nt,1) if nt else 0
        nc,nr = s["n_conf"],s["n_corr"]

        chips = ""
        if nv>0:
            chips += f'<span class="chip conf">✓ {nc} confirmed</span>'
            chips += f'<span class="chip corr">✏ {nr} corrected</span>'
        else:
            chips = '<span class="chip pend">Not started</span>'

        bars = ""
        for i,(m,h,c) in enumerate(zip(s["model_pct"],s["human_pct"],CCOLORS)):
            hbar = f'<div class="bh" style="width:{h}%;background:{c}"></div>' if nv>0 else ""
            bars += f'<div class="br"><div class="bl">{CSHORT[i]}</div><div class="bt"><div class="bm" style="width:{m}%;background:{c}"></div>{hbar}</div><div class="bv">{m:.0f}%</div></div>'

        color = "var(--green)" if is_k else "var(--accent)"
        mv_json = json.dumps(s["model_pct"])
        hv_json = json.dumps(s["human_pct"])
        cards += f"""<div class="card {badge}" data-sid="{sid}" data-mv='{mv_json}' data-hv='{hv_json}'>
          <div class="ctop"><div><div class="cname">{slide}</div></div><div class="cbadge {badge}">{lbl}</div></div>
          <div class="prow"><div class="plbl">Progress</div><div class="pcount">{nv}/{nt} ({pct:.0f}%)</div></div>
          <div class="ptrack"><div class="pfill" style="width:{pct}%;background:{color}"></div></div>
          <div class="chips">{chips}</div>
          <div class="radars">
            <div class="rb"><canvas id="rm{sid}"></canvas><div class="rl m">▬ Model (all {nt})</div></div>
            <div class="rvs">vs</div>
            <div class="rb"><canvas id="rh{sid}"></canvas><div class="rl h">▬ Human ({nv} val.)</div></div>
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:.58em;color:var(--muted);margin:8px 0 4px;display:flex;justify-content:space-between"><span>Class</span><span style="color:rgba(126,232,250,.5)">▬ model</span><span style="color:rgba(82,196,26,.8)">▬ human</span><span>%</span></div>
          <div class="bars">{bars}</div>
        </div>"""

    html = RESULTS_HTML
    for k,v in [("__TIMESTAMP__",ts),("__TOTAL_VAL__",str(total_validated)),
                ("__TOTAL_P__",str(total_patches)),("__PCT_DONE__",str(pct_done)),
                ("__PCT_CONF__",str(pct_conf)),("__PCT_CORR__",str(pct_corr)),
                ("__N_CONF__",str(n_confirmed)),("__N_CORR__",str(n_corrected)),
                ("__N_STARTED__",str(n_started)),("__N_SLIDES__",str(n_slides)),
                ("__N_DONE__",str(n_done)),("__CARDS__",cards)]:
        html = html.replace(k,v)
    return html

if __name__ == '__main__':
    p=get_patches(); l=load_validated(); pr=load_predictions()
    print(f"✓ {len(p)} patches | {len(pr)} predictions | {len(l)} validated")
    import socket
    ip=socket.gethostbyname(socket.gethostname())
    print(f"→ Validation: http://localhost:5000")
    print(f"→ Results:    http://localhost:5000/results")
    print(f"→ Network:    http://{ip}:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
