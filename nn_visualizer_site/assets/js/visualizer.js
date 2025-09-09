// ===================================================
// Neural Network Visualizer — JS (Comments in English)
// ===================================================

// ---------- Presets ----------
const PRESETS = {
  roadvision: {
    name: "Road_Vision",
    nodes: [
      { id:"input",  type:"Input",  shape:"(T×5×224×224)", note:"RGB+edge+mask (5‑ch)" },
      { id:"cnn",    type:"CNN",    shape:"(T×C'×H'×W')", note:"MobileNetV3 feature maps" },
      { id:"gap",    type:"GAP",    shape:"(T×C')", note:"global average pooling" },
      { id:"gru",    type:"GRU",    shape:"(T×H)", note:"temporal modeling" },
      { id:"mlp",    type:"MLP",    shape:"(T×K)", note:"classifier" },
      { id:"softmax",type:"Softmax",shape:"(T×K)", note:"EMA+Hysteresis downstream" }
    ],
    edges: [
      ["input","cnn"],["cnn","gap"],["gap","gru"],["gru","mlp"],["mlp","softmax"]
    ]
  },
  mlp: {
    name: "MLP",
    nodes:[
      {id:"x", type:"Input", shape:"(N×D)"},
      {id:"h1", type:"Dense+ReLU", shape:"(N×128)"},
      {id:"h2", type:"Dense+ReLU", shape:"(N×64)"},
      {id:"y", type:"Dense+Softmax", shape:"(N×K)"}
    ],
    edges:[["x","h1"],["h1","h2"],["h2","y"]]
  },
  cnn: {
    name:"Simple CNN",
    nodes:[
      {id:"img", type:"Input", shape:"(3×224×224)"},
      {id:"c1", type:"Conv3×3+BN+ReLU", shape:"(32×112×112)"},
      {id:"p1", type:"MaxPool2×2", shape:"(32×56×56)"},
      {id:"c2", type:"Conv3×3+BN+ReLU", shape:"(64×56×56)"},
      {id:"gap", type:"GAP", shape:"(64)"},
      {id:"fc", type:"Dense+Softmax", shape:"(K)"}
    ],
    edges:[["img","c1"],["c1","p1"],["p1","c2"],["c2","gap"],["gap","fc"]]
  },
  vit: {
    name:"ViT (simplified)",
    nodes:[
      {id:"img", type:"Image", shape:"(3×224×224)"},
      {id:"patch", type:"Patchify+Linear", shape:"(N_p×D)"},
      {id:"enc", type:"TransformerEncoder×L", shape:"(N_p×D)"},
      {id:"cls", type:"[CLS] head", shape:"(D)"},
      {id:"out", type:"Dense+Softmax", shape:"(K)"}
    ],
    edges:[["img","patch"],["patch","enc"],["enc","cls"],["cls","out"]]
  },
  auto: {
    name:"Autoencoder",
    nodes:[
      {id:"x", type:"Input", shape:"(N×D)"},
      {id:"enc1", type:"Dense+ReLU", shape:"(N×128)"},
      {id:"z", type:"Dense", shape:"(N×32)"},
      {id:"dec1", type:"Dense+ReLU", shape:"(N×128)"},
      {id:"xhat", type:"Dense+Sigmoid", shape:"(N×D)"}
    ],
    edges:[["x","enc1"],["enc1","z"],["z","dec1"],["dec1","xhat"]]
  }
};

// ---------- DOM ----------
const svg = document.getElementById('svg');
const tooltip = document.getElementById('tooltip');
const kv = document.getElementById('kv');
const layerTitle = document.getElementById('layerTitle');
const presetSel = document.getElementById('preset');
const ta = document.getElementById('modelJson');

// ---------- Rendering ----------
function clearSVG(){ while(svg.firstChild) svg.removeChild(svg.firstChild); }

function layoutColumns(model){
  // Assign columns by simple topological order from edges (Kahn-like layering)
  const indeg = {}; model.nodes.forEach(n=>indeg[n.id]=0);
  model.edges.forEach(([u,v])=> indeg[v] = (indeg[v]||0)+1);
  const layers = []; const seen = new Set();
  let frontier = model.nodes.filter(n=>indeg[n.id]===0).map(n=>n.id);
  let col = 0;
  while(frontier.length){
    layers[col] = [];
    const next = [];
    frontier.forEach(id => { if(!seen.has(id)){ seen.add(id); layers[col].push(id); }});
    model.edges.forEach(([u,v])=>{
      if(seen.has(u) && !seen.has(v)){
        const parents = model.edges.filter(e=>e[1]===v).map(e=>e[0]);
        if(parents.every(p=>seen.has(p))) next.push(v);
      }
    });
    frontier = [...new Set(next)];
    col++;
  }
  // Fallback for isolated nodes
  model.nodes.forEach(n=>{ if(!layers.flat().includes(n.id)){ (layers[0]||(layers[0]=[])).push(n.id); }});
  return layers;
}

function renderModel(model){
  clearSVG();
  const W=1200, H=675; const padX=80, padY=70;
  const layers = layoutColumns(model);
  const colWidth = layers.length>1 ? (W-2*padX)/(layers.length-1) : 0;
  const nodePos = {}; // id -> {x,y}

  // Compute positions per layer
  layers.forEach((ids, i)=>{
    const n = ids.length; const totalHeight = H-2*padY; const step = n>1? totalHeight/(n-1):0; const x = padX + i*colWidth;
    ids.forEach((id, j)=>{ nodePos[id] = {x, y: padY + j*step}; });
  });

  // Draw edges
  model.edges.forEach(([u,v])=>{
    const a=nodePos[u], b=nodePos[v];
    const dx = (b.x-a.x)*0.6; // control offset for smooth curve
    const d = `M ${a.x} ${a.y} C ${a.x+dx} ${a.y}, ${b.x-dx} ${b.y}, ${b.x} ${b.y}`;
    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d', d);
    path.setAttribute('class','edge');
    path.dataset.u=u; path.dataset.v=v;
    svg.appendChild(path);
  });

  // Draw nodes
  model.nodes.forEach(n=>{
    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.setAttribute('class','node'); g.dataset.id = n.id;
    const {x,y}=nodePos[n.id];
    // rounded rectangle node
    const w=160, h=54, rx=10; const xx=x-w/2, yy=y-h/2;
    const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x', xx); rect.setAttribute('y', yy); rect.setAttribute('width', w); rect.setAttribute('height', h); rect.setAttribute('rx', rx);
    g.appendChild(rect);
    // title
    const t1 = document.createElementNS('http://www.w3.org/2000/svg','text');
    t1.setAttribute('x', x); t1.setAttribute('y', y-4); t1.setAttribute('text-anchor','middle'); t1.textContent = n.type;
    g.appendChild(t1);
    // shape
    const t2 = document.createElementNS('http://www.w3.org/2000/svg','text');
    t2.setAttribute('x', x); t2.setAttribute('y', y+14); t2.setAttribute('text-anchor','middle'); t2.setAttribute('class','small'); t2.textContent = n.shape||'';
    g.appendChild(t2);

    // Hover behavior
    g.addEventListener('mousemove', (e)=>{
      highlight(n.id);
      showInfo(n);
      showTooltip(e.clientX, e.clientY, `${n.type}${n.shape ? "\n"+n.shape : ""}${n.note ? "\n"+n.note : ""}`);
    });
    g.addEventListener('mouseleave', ()=>{
      unhighlight(); hideTooltip();
    });

    svg.appendChild(g);
  });

  function highlight(id){
    unhighlight();
    [...svg.querySelectorAll('.edge')].forEach(p=>{
      if(p.dataset.u===id || p.dataset.v===id){ p.classList.add('edge-hi'); }
    });
  }
  function unhighlight(){
    [...svg.querySelectorAll('.edge-hi')].forEach(p=>p.classList.remove('edge-hi'));
  }
}

function showInfo(n){
  layerTitle.textContent = `${n.type} — ${n.id}`;
  kv.innerHTML = '';
  const rows = [
    ['Shape', n.shape||'-'],
    ['Note', n.note||'-']
  ];
  for(const [k,v] of rows){
    const a = document.createElement('div'); a.textContent=k; kv.appendChild(a);
    const b = document.createElement('div'); b.textContent=v; kv.appendChild(b);
  }
}

function showTooltip(x,y,msg){
  const t=tooltip; t.textContent=msg; t.style.left=x+'px'; t.style.top=y+'px'; t.style.opacity=1;
}
function hideTooltip(){ tooltip.style.opacity=0; }

// ---------- Animation (forward pulse) ----------
function animateForward(model){
  const edges = [...svg.querySelectorAll('.edge')];
  let i=0; const timer=setInterval(()=>{
    if(i>=edges.length){ clearInterval(timer); return; }
    edges[i].classList.add('edge-hi');
    setTimeout(()=>edges[i].classList.remove('edge-hi'), 600);
    i++;
  }, 80);
}

// ---------- JSON editing ----------
function setPreset(key){ ta.value = JSON.stringify(PRESETS[key], null, 2); parseAndRender(); }

function parseAndRender(){
  try{ const model = JSON.parse(ta.value); renderModel(model); } catch(e){ /* ignore */ }
}

// Bindings
document.getElementById('render').onclick = parseAndRender;
document.getElementById('pretty').onclick = ()=>{ try{ ta.value = JSON.stringify(JSON.parse(ta.value), null, 2); }catch(e){} };
document.getElementById('animate').onclick = ()=>{ try{ animateForward(JSON.parse(ta.value)); }catch(e){} };
document.getElementById('reset').onclick = ()=> setPreset(presetSel.value);
presetSel.onchange = ()=> setPreset(presetSel.value);

// Init
setPreset('roadvision');
