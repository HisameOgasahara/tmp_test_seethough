"""Step 4: stack per-layer SVG groups into one SVG and an animated HTML (GSAP).

Each layer keeps See-through's draw order and gets three nested groups:
    <g class="head?">        whole-head bob (applied to every head-related layer with the same pivot)
      <g data-role="...">    role motion: hair sway, blink, breathing
        <g transform=...>    upscaled-layer coords -> canvas coords
Cel-style blink: if closed-eye layers are supplied they are swapped in; otherwise the
open eyes are squashed vertically as a placeholder.
"""
import json

ROLE_RULES = [  # first match wins; order matters ("topwear" contains "ear", "eyewear" contains "eye")
    ("hair_front", ["front hair", "hairf"]),
    ("hair_back", ["back hair", "hairb"]),
    ("accessory", ["headwear", "eyewear", "earwear", "neckwear", "objects"]),
    ("body", ["topwear", "handwear", "bottomwear", "legwear", "footwear", "tail", "wings"]),
    ("eyes", ["eye", "irides", "iris", "lash"]),
    ("brows", ["brow"]),
    ("mouth", ["mouth"]),
    ("head", ["head", "face", "nose", "neck", "ears", "earl", "earr"]),
]
HEAD_ROLES = {"hair_front", "hair_back", "eyes", "brows", "mouth", "head"}


def role_of(name):
    n = name.lower()
    for role, keys in ROLE_RULES:
        if any(k in n for k in keys):
            return role
    return "other"


def layer_group(name, body, offset, scale, role, extra_class=""):
    x0, y0 = offset
    classes = " ".join(c for c in ["head" if role in HEAD_ROLES or "headwear" in name.lower() else "",
                                   extra_class] if c)
    safe = name.replace('"', "")
    return (f'<g class="{classes}" data-name="{safe}"><g data-role="{role}">'
            f'<g transform="translate({x0} {y0}) scale({1/scale:.6f})">{body}</g></g></g>')


def build_svg(groups, box):
    x0, y0, x1, y1 = box
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {x1-x0} {y1-y0}" '
            f'id="char"><g id="breath">{"".join(groups)}</g></svg>')


HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cel Puppet</title>
<style>
:root{--bg:#f4f4f2;--fg:#222}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#eee}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#eee}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui;display:flex;flex-direction:column;align-items:center;padding:16px}
#stage{width:min(100%,720px)} #stage svg{width:100%;height:auto;display:block}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid #8884;background:transparent;color:inherit;cursor:pointer}
</style></head><body>
<div id="stage">__SVG__</div>
<div class="bar"><button id="play">일시정지</button><button id="blink">깜빡</button></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
const P = __PARAMS__;
const svg = document.getElementById('char');
const q = s => Array.from(svg.querySelectorAll(s));
const uBox = els => { // bbox in root coords
  let b=null; els.forEach(e=>{ const r=e.getBBox(), m=e.getCTM(), root=svg.getCTM().inverse();
    const mm=root.multiply(m); const pts=[[r.x,r.y],[r.x+r.width,r.y],[r.x,r.y+r.height],[r.x+r.width,r.y+r.height]]
      .map(([x,y])=>({x:mm.a*x+mm.c*y+mm.e,y:mm.b*x+mm.d*y+mm.f}));
    pts.forEach(p=>{ b=b?{x:Math.min(b.x,p.x),y:Math.min(b.y,p.y),X:Math.max(b.X,p.x),Y:Math.max(b.Y,p.y)}:{x:p.x,y:p.y,X:p.x,Y:p.y}; });
  }); return b; };
const tl = gsap.timeline({repeat:-1});
const all = [];

// breathing: whole body, pivot at the bottom center
const vb = svg.viewBox.baseVal;
all.push(gsap.to(q('#breath'), {scaleY:1+P.breath, svgOrigin:`${vb.x+vb.width/2} ${vb.y+vb.height}`,
  duration:P.breathSec/2, yoyo:true, repeat:-1, ease:'sine.inOut'}));

// head bob: every head layer around the neck (bottom center of head-role layers)
const heads = q('g.head'); const hb = uBox(q('g.head [data-role="head"]').length? q('g.head [data-role="head"]'):heads);
if (heads.length && hb) all.push(gsap.to(heads, {rotation:P.headDeg, svgOrigin:`${(hb.x+hb.X)/2} ${hb.Y}`,
  duration:P.headSec/2, yoyo:true, repeat:-1, ease:'sine.inOut'}));

// hair sway: rotate from the roots (top center), back hair lags behind front hair
[['hair_front',0],['hair_back',0.35]].forEach(([r,delay])=>{
  q(`[data-role="${r}"]`).forEach(g=>{ const b=uBox([g]); if(!b) return;
    all.push(gsap.fromTo(g,{rotation:-P.hairDeg},{rotation:P.hairDeg, svgOrigin:`${(b.x+b.X)/2} ${b.y}`,
      duration:P.hairSec/2, yoyo:true, repeat:-1, ease:'sine.inOut', delay})); });
});

// blink: swap to closed-eye cels if present, else squash open eyes
const open = q('[data-role="eyes"]').filter(g=>!g.closest('.closed'));
const closed = q('g.closed'); gsap.set(closed,{autoAlpha:0});
function blink(){
  const t = gsap.timeline();
  if (closed.length){ t.set(open,{autoAlpha:0}).set(closed,{autoAlpha:1}).set(open,{autoAlpha:1},P.blinkHold).set(closed,{autoAlpha:0},P.blinkHold); }
  else { const b=uBox(open); if(!b) return t; t.to(open,{scaleY:0.08, svgOrigin:`${(b.x+b.X)/2} ${b.y+(b.Y-b.y)*0.6}`, duration:0.06})
           .to(open,{scaleY:1, duration:0.08}, `+=${P.blinkHold}`); }
  return t;
}
(function loop(){ gsap.delayedCall(P.blinkMin + Math.random()*(P.blinkMax-P.blinkMin), ()=>{ blink(); loop(); }); })();

let playing = true;
document.getElementById('play').onclick = e => { playing=!playing; gsap.globalTimeline[playing?'resume':'pause']();
  e.target.textContent = playing?'일시정지':'재생'; };
document.getElementById('blink').onclick = () => blink();
</script></body></html>"""

DEFAULT_PARAMS = dict(breath=0.006, breathSec=3.6, headDeg=0.8, headSec=4.2, hairDeg=1.2, hairSec=2.8,
                      blinkMin=2.5, blinkMax=5.5, blinkHold=0.09)


def build_html(svg_text, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return HTML.replace("__SVG__", svg_text).replace("__PARAMS__", json.dumps(p))
