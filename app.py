import os
from functools import lru_cache
from html import escape

import gradio as gr
import pandas as pd
from google import genai
from google.genai import types
from joblib import load

# ==========================================
# CONFIG
# ==========================================
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
client = genai.Client(api_key="AQ.Ab8RN6Jy8F9q-Y8_PBFaerPbxrW_-9bYlgT8o5puvE0HPsE5wA")
# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

df = pd.read_csv("./dataset/training_data.csv")
symptoms = list(df.columns[:-2])
symptom_index = {s: i for i, s in enumerate(symptoms)}
model = load("./saved_model/random_forest.joblib")

pretty = lambda s: s.replace("_", " ").strip().title()
choices = [(pretty(s), s) for s in symptoms]

COMMON = ["fever", "high_fever", "cough", "headache", "fatigue", "vomiting", "nausea",
          "chest_pain", "skin_rash", "itching", "stomach_pain", "joint_pain", "dizziness",
          "breathlessness", "sore_throat", "runny_nose"]
quick = [s for s in COMMON if s in symptom_index][:10]

CHAT_SYSTEM = (
    "You are My AMIGO, a warm, calm, empathetic AI health companion. "
    "Use simple language, keep answers short, never diagnose, "
    "and suggest seeing a doctor when it is appropriate."
)

# ==========================================
# HTML SNIPPETS
# ==========================================
HERO = f"""
<div class="hero">
  <div class="hero-badge">💚 Your AI health companion</div>
  <h1>Feeling off? Let’s figure it out together.</h1>
  <p>Pick your symptoms for an instant AI insight, or just chat with AMIGO about how you feel.</p>
  <div class="hero-pills"><span>⚡ Instant insights</span><span>🩺 {len(symptoms)} symptoms</span><span>💬 Friendly chat</span></div>
  <div class="sticker s1">🩺</div><div class="sticker s2">💊</div><div class="sticker s3 beat">❤️</div>
  <svg class="ecg" viewBox="0 0 1200 120" preserveAspectRatio="none">
    <path d="M0 70H330L350 70L370 30L395 105L420 5L445 100L465 70H760L780 70L800 42L820 95L840 70H1200"/>
  </svg>
</div>"""


def notice(text):
    return f'<div class="res"><div class="note">{escape(text)}</div></div>'


def loading(n):
    return f"""
    <div class="res"><div class="loading">
        <div class="orb"><span></span><span></span><span></span></div>
        <p class="load-text">Analyzing {n} symptom{'s' if n != 1 else ''}…</p>
        <div class="skel"></div><div class="skel short"></div><div class="skel"></div>
    </div></div>"""


@lru_cache(maxsize=128)
def explain(disease: str) -> str:
    prompt = f"""Disease: {disease}
Return ONLY raw HTML (no markdown, no code fences, no html/body/style tags), concise,
in exactly this structure, 3-4 short <li> each:

<section class="ins spot tilt"><h3>🌿 Possible causes</h3><ul><li>..</li></ul></section>
<section class="ins spot tilt"><h3>🍵 How to recover</h3><ul><li>..</li></ul></section>
<section class="ins spot tilt"><h3>🩺 When to see a doctor</h3><ul><li>..</li></ul></section>
<section class="ins spot tilt"><h3>🛡️ Prevention tips</h3><ul><li>..</li></ul></section>"""
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = (resp.text or "").replace("```html", "").replace("```", "").strip()
    return text or '<section class="ins"><h3>Insights</h3><p>No extra insights available.</p></section>'


def top_predictions(vector, k=3):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([vector])[0]
        order = probs.argsort()[::-1][:k]
        return [(str(model.classes_[i]), float(probs[i]) * 100) for i in order if probs[i] > 0]
    return [(str(model.predict([vector])[0]), 100.0)]


# ==========================================
# SYMPTOM CHECKER
# ==========================================
def predict_disease(selected):
    if not selected:
        yield notice("Pick at least one symptom to get started.")
        return
    try:
        yield loading(len(selected))
        vector = [0] * len(symptoms)
        for s in selected:
            if s in symptom_index:
                vector[symptom_index[s]] = 1

        preds = top_predictions(vector)
        best = preds[0][0]
        bars = "".join(
            f'<div class="bar"><span>{escape(n)}</span>'
            f'<div class="track"><i style="--w:{p:.0f}%"></i></div><b>{p:.0f}%</b></div>'
            for n, p in preds
        )
        chips = "".join(f"<em>{escape(pretty(s))}</em>" for s in selected)

        yield f"""
        <div class="res">
            <div class="hero-result spot tilt">
                <p class="k">Closest match</p>
                <h2 class="dz">{escape(best)}</h2>
                <div class="chips">{chips}</div>
                <div class="bars">{bars}</div>
            </div>
            <div class="note">⚠️ AI-assisted insight only. This is not a medical diagnosis.</div>
            <div class="ins-grid">{explain(best)}</div>
        </div>"""
    except Exception as e:
        yield notice(f"Something went wrong: {e}")


# ==========================================
# CHAT
# ==========================================
def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def chat_with_amigo(message, history):
    try:
        contents = []
        for h in (history or [])[-12:]:
            if isinstance(h, dict):
                role = "model" if h.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": _text(h.get("content"))}]})
            else:
                u, b = h
                if u: contents.append({"role": "user", "parts": [{"text": _text(u)}]})
                if b: contents.append({"role": "model", "parts": [{"text": _text(b)}]})
        contents.append({"role": "user", "parts": [{"text": message}]})

        acc = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents,
            config=types.GenerateContentConfig(system_instruction=CHAT_SYSTEM),
        ):
            acc += chunk.text or ""
            yield acc
        if not acc:
            yield "I'm here with you. Could you try asking again?"
    except Exception as e:
        yield f"Couldn't reach Gemini: {e}"


# ==========================================
# HEAD: fonts + animated background + mouse magic
# ==========================================
head = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script>
(() => {
  if (window.__amigo) return; window.__amigo = true;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- background layer: blobs + floating health icons + cursor glow ----
  const bg = document.createElement('div'); bg.id = 'bg';
  const icons = ['❤️','💊','🩺','🧬','🍎','💧','🌿','🧠','➕','🫀','🥦','😴'];
  bg.innerHTML = '<i class="blob b1"></i><i class="blob b2"></i><i class="blob b3"></i><i id="glow"></i>' +
    icons.map((e, i) => {
      const l = (i * 37 + 5) % 92, t = (i * 53 + 8) % 84, s = 26 + (i * 13) % 30, d = 10 + (i * 7) % 24;
      return `<span class="fi" style="left:${l}%;top:${t}%;--d:${d};font-size:${s}px"><b style="animation-duration:${7 + i % 5 * 2}s;animation-delay:-${i * 1.3}s">${e}</b></span>`;
    }).join('');
  document.body.prepend(bg);
  const glow = bg.querySelector('#glow');
  let tx = innerWidth / 2, ty = innerHeight / 3, x = tx, y = ty;
  (function loop() {
    x += (tx - x) * .1; y += (ty - y) * .1;
    glow.style.transform = `translate3d(${x - 250}px,${y - 250}px,0)`;
    requestAnimationFrame(loop);
  })();

  // ---- pointer interactions (delegated) ----
  addEventListener('pointermove', e => {
    tx = e.clientX; ty = e.clientY;
    bg.style.setProperty('--px', (e.clientX / innerWidth - .5) * 2);
    bg.style.setProperty('--py', (e.clientY / innerHeight - .5) * 2);
    const c = e.target.closest?.('.spot');
    if (c) { const r = c.getBoundingClientRect();
      c.style.setProperty('--x', (e.clientX - r.left) + 'px');
      c.style.setProperty('--y', (e.clientY - r.top) + 'px'); }
    if (still) return;
    const t = e.target.closest?.('.tilt');
    if (t) { const r = t.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width - .5, py = (e.clientY - r.top) / r.height - .5;
      t.style.transform = `perspective(900px) rotateX(${-py * 7}deg) rotateY(${px * 9}deg) translateY(-3px)`; }
    const b = e.target.closest?.('button.magnet');
    if (b) { const r = b.getBoundingClientRect();
      b.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * .14}px,${(e.clientY - r.top - r.height / 2) * .24}px)`; }
  }, { passive: true });

  addEventListener('pointerout', e => {
    for (const sel of ['.tilt', 'button.magnet']) {
      const el = e.target.closest?.(sel);
      if (el && !el.contains(e.relatedTarget)) el.style.transform = '';
    }
  });

  addEventListener('pointerdown', e => {
    const b = e.target.closest?.('button.magnet, button.pill, button.ghost'); if (!b) return;
    const r = b.getBoundingClientRect(), s = document.createElement('span');
    s.className = 'ripple';
    s.style.cssText = `left:${e.clientX - r.left}px;top:${e.clientY - r.top}px`;
    b.appendChild(s); setTimeout(() => s.remove(), 650);
  });

  // ---- confetti when a result appears ----
  const burst = () => {
    if (still) return;
    for (let i = 0; i < 24; i++) {
      const s = document.createElement('span');
      s.textContent = ['💚','✨','💜','🩵','➕'][i % 5];
      s.style.cssText = `position:fixed;left:50%;top:42%;font-size:${14 + Math.random() * 16}px;pointer-events:none;z-index:9999`;
      document.body.appendChild(s);
      const a = Math.random() * 6.28, d = 140 + Math.random() * 260;
      s.animate([
        { transform: 'translate(-50%,-50%) scale(.3)', opacity: 1 },
        { transform: `translate(${Math.cos(a) * d}px,${Math.sin(a) * d + 90}px) rotate(${Math.random() * 360}deg) scale(1)`, opacity: 0 }
      ], { duration: 1100 + Math.random() * 500, easing: 'cubic-bezier(.2,.8,.3,1)' }).onfinish = () => s.remove();
    }
  };
  new MutationObserver(ms => {
    for (const m of ms) for (const n of m.addedNodes)
      if (n.nodeType === 1 && (n.matches?.('.hero-result') || n.querySelector?.('.hero-result'))) return burst();
  }).observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

# ==========================================
# CSS
# ==========================================
css = """
:root{--ink:#1d1b3f;--muted:#5f6788;--violet:#7c3aed;--teal:#14b8a6;--sky:#0ea5e9;--line:rgba(29,27,63,.1)}
html,body,gradio-app{background:linear-gradient(135deg,#f4ecff 0%,#e2f8f5 50%,#fff0e8 100%) fixed!important}
.gradio-container{
  --body-background-fill:transparent; --background-fill-primary:transparent;
  --background-fill-secondary:rgba(255,255,255,.7); --block-background-fill:transparent;
  --body-text-color:var(--ink); --body-text-color-subdued:var(--muted);
  --block-label-text-color:var(--muted); --block-title-text-color:var(--ink);
  --border-color-primary:var(--line); --input-background-fill:#fff;
  --input-border-color:rgba(124,58,237,.2); --input-border-color-focus:var(--violet);
  --color-accent:var(--violet); --color-accent-soft:rgba(124,58,237,.1);
  --block-border-width:0px; --block-shadow:none; --shadow-drop:none; --block-radius:22px;
  background:transparent!important; position:relative; z-index:1; color:var(--ink);
  font-family:'Plus Jakarta Sans',system-ui,sans-serif!important; max-width:1080px!important; margin:auto;
}
footer{display:none!important}

/* ---------- animated background ---------- */
#bg{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;
  background-image:radial-gradient(rgba(124,58,237,.14) 1.2px,transparent 1.4px);background-size:30px 30px}
.blob{position:absolute;width:46vmax;height:46vmax;border-radius:50%;filter:blur(70px);opacity:.55;animation:blob 18s ease-in-out infinite alternate}
.b1{background:#a78bfa;left:-12vmax;top:-14vmax}
.b2{background:#5eead4;right:-14vmax;top:18vh;animation-delay:-6s}
.b3{background:#fda4af;left:25vw;bottom:-26vmax;animation-delay:-11s;opacity:.45}
@keyframes blob{to{transform:translate3d(6vmax,5vmax,0) scale(1.18) rotate(30deg)}}
#glow{position:absolute;top:0;left:0;width:500px;height:500px;border-radius:50%;will-change:transform;
  background:radial-gradient(circle,rgba(124,58,237,.2),rgba(20,184,166,.1) 45%,transparent 68%)}
.fi{position:absolute;opacity:.55;transition:transform .5s ease-out;
  transform:translate(calc(var(--px,0)*var(--d)*2px),calc(var(--py,0)*var(--d)*2px))}
.fi b{display:block;font-weight:400;animation:bob 8s ease-in-out infinite alternate;filter:drop-shadow(0 8px 10px rgba(29,27,63,.15))}
@keyframes bob{from{transform:translateY(-14px) rotate(-8deg)}to{transform:translateY(14px) rotate(8deg)}}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;margin:26px 0 22px;padding:60px 48px 84px;border-radius:38px;color:#fff;
  background:linear-gradient(120deg,#7c3aed,#4f46e5 35%,#0ea5e9 70%,#14b8a6);background-size:240% 240%;
  animation:flow 14s ease infinite,rise .8s cubic-bezier(.2,.8,.2,1) both;
  box-shadow:0 34px 80px -34px rgba(79,70,229,.7)}
.hero::before{content:"";position:absolute;width:420px;height:420px;right:-90px;top:-120px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.35),transparent 65%)}
@keyframes flow{50%{background-position:100% 60%}}
@keyframes rise{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
.hero-badge{display:inline-block;padding:8px 18px;border-radius:99px;background:rgba(255,255,255,.2);
  border:1px solid rgba(255,255,255,.35);backdrop-filter:blur(10px);font-weight:600;font-size:14px}
.hero h1{position:relative;font-size:clamp(34px,5.4vw,60px);font-weight:800;line-height:1.05;letter-spacing:-.035em;
  margin:20px 0 0;max-width:15ch;color:#fff!important}
.hero p{position:relative;max-width:46ch;font-size:18px;line-height:1.6;margin:16px 0 0;color:rgba(255,255,255,.9)}
.hero-pills{position:relative;display:flex;flex-wrap:wrap;gap:10px;margin-top:26px}
.hero-pills span{padding:9px 16px;border-radius:99px;background:#fff;color:var(--ink);font-weight:600;font-size:14px;
  box-shadow:0 8px 20px -8px rgba(0,0,0,.3);transition:transform .25s cubic-bezier(.3,1.6,.5,1)}
.hero-pills span:hover{transform:translateY(-4px) rotate(-2deg)}
.sticker{position:absolute;display:grid;place-items:center;border-radius:28px;background:rgba(255,255,255,.22);
  border:1px solid rgba(255,255,255,.45);backdrop-filter:blur(12px);box-shadow:0 20px 40px -16px rgba(0,0,0,.35);
  animation:bob 6s ease-in-out infinite alternate}
.s1{right:9%;top:12%;width:104px;height:104px;font-size:54px;rotate:8deg}
.s2{right:26%;top:52%;width:76px;height:76px;font-size:38px;rotate:-10deg;animation-delay:-2s}
.s3{right:4%;bottom:26%;width:84px;height:84px;font-size:44px;rotate:-6deg;animation-delay:-4s}
.beat{animation:bob 6s ease-in-out infinite alternate,beat 1.2s ease-in-out infinite}
@keyframes beat{0%,100%{scale:1}15%{scale:1.15}30%{scale:1}45%{scale:1.2}}
.ecg{position:absolute;left:0;bottom:6px;width:100%;height:76px}
.ecg path{fill:none;stroke:#fff;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;opacity:.75;
  stroke-dasharray:1500;stroke-dashoffset:1500;animation:ecg 4.5s linear infinite;filter:drop-shadow(0 0 6px #fff)}
@keyframes ecg{60%{stroke-dashoffset:0}100%{stroke-dashoffset:-1500}}

/* ---------- panels ---------- */
.ins,.loading{position:relative;overflow:hidden;border-radius:28px;background:rgba(255,255,255,.78);
  border:1px solid rgba(255,255,255,.9);backdrop-filter:blur(18px) saturate(150%);
  box-shadow:0 24px 60px -28px rgba(124,58,237,.35)}
.panel{position:relative;z-index:30;overflow:visible!important;padding:30px!important;border-radius:28px;
  background:rgba(255,255,255,.88);border:1px solid #fff;box-shadow:0 24px 60px -28px rgba(124,58,237,.35)}
.spot::after{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;opacity:0;transition:opacity .3s;
  background:radial-gradient(320px circle at var(--x,50%) var(--y,50%),rgba(124,58,237,.12),transparent 65%)}
.spot:hover::after{opacity:1}
.tilt{transition:transform .25s ease-out;will-change:transform}
.panel-title{font-size:28px;font-weight:800;letter-spacing:-.02em;margin:0;color:var(--ink)}
.panel-sub{color:var(--muted);margin:6px 0 0;font-size:15px}

/* ---------- inputs ---------- */
.gradio-container input,.gradio-container textarea{color:var(--ink)!important;font-family:inherit!important}
.gradio-container .wrap{border-radius:16px!important}
.gradio-container .token{background:linear-gradient(135deg,#ede4ff,#d9f7f3)!important;color:#4c1d95!important;font-weight:600;
  border:1px solid rgba(124,58,237,.25)!important;border-radius:99px!important;animation:pop .3s cubic-bezier(.3,1.7,.5,1)}
@keyframes pop{from{transform:scale(.5);opacity:0}}
ul.options,[role="listbox"]{background:#fff!important;opacity:1!important;z-index:99999!important;
  border:1px solid var(--line)!important;border-radius:16px!important;
  box-shadow:0 24px 50px -16px rgba(29,27,63,.3)!important}
ul.options li,[role="listbox"] li,[role="option"]{background:#fff;color:var(--ink)!important}
ul.options li:hover,ul.options li.selected,[role="option"]:hover,[role="option"][aria-selected="true"]{background:#f1e9ff!important}

/* ---------- quick-pick pills ---------- */
.pills{flex-wrap:wrap!important;gap:10px!important}
.pills button{flex:0 0 auto!important;min-width:0!important;width:auto!important}
button.pill{position:relative;overflow:hidden;padding:8px 16px!important;border-radius:99px!important;background:#fff!important;
  color:var(--ink)!important;font-weight:600!important;border:1.5px solid rgba(124,58,237,.2)!important;
  transition:transform .25s cubic-bezier(.3,1.6,.5,1),background .2s,color .2s,box-shadow .2s!important}
button.pill:hover{transform:translateY(-3px) rotate(-1.5deg);background:linear-gradient(135deg,#7c3aed,#14b8a6)!important;
  color:#fff!important;box-shadow:0 12px 22px -10px rgba(124,58,237,.6)}

/* ---------- buttons ---------- */
button.magnet,button.ghost{position:relative;overflow:hidden;border-radius:99px!important;font-weight:700!important;font-size:16px!important;
  transition:transform .25s cubic-bezier(.2,.8,.2,1),box-shadow .3s!important}
button.magnet{background:linear-gradient(120deg,#7c3aed,#0ea5e9,#14b8a6)!important;background-size:200% 100%!important;color:#fff!important;
  border:0!important;box-shadow:0 14px 30px -10px rgba(124,58,237,.7);animation:flow 6s ease infinite}
button.magnet:hover{box-shadow:0 20px 44px -10px rgba(14,165,233,.8)}
button.magnet:active{scale:.96}
button.ghost{background:#fff!important;color:var(--muted)!important;border:1.5px solid var(--line)!important}
button.ghost:hover{color:var(--ink)!important;transform:translateY(-2px)}
.ripple{position:absolute;width:12px;height:12px;margin:-6px;border-radius:50%;background:rgba(255,255,255,.65);
  pointer-events:none;animation:ripple .65s ease-out forwards}
button.pill .ripple,button.ghost .ripple{background:rgba(124,58,237,.3)}
@keyframes ripple{to{transform:scale(28);opacity:0}}

/* ---------- tabs: segmented pill ---------- */
.tab-container{display:inline-flex!important;gap:4px;padding:6px!important;border-radius:99px;height:auto!important;
  background:rgba(255,255,255,.75);border:1px solid #fff;box-shadow:0 10px 30px -14px rgba(124,58,237,.35)}
.tab-container::after,button[role="tab"]::after{display:none!important}
.tab-wrapper{border:0!important;height:auto!important;margin-bottom:12px}
button[role="tab"]{padding:10px 22px!important;border-radius:99px!important;border:0!important;color:var(--muted)!important;
  font-weight:700!important;transition:all .3s cubic-bezier(.2,.8,.2,1)!important}
button[role="tab"]:hover{color:var(--ink)!important}
button[role="tab"][aria-selected="true"]{color:#fff!important;background:linear-gradient(120deg,#7c3aed,#0ea5e9)!important;
  box-shadow:0 10px 20px -8px rgba(124,58,237,.7)}

/* ---------- results ---------- */
.res{display:grid;gap:18px;margin-top:20px}
.res>*{animation:rise .6s cubic-bezier(.2,.8,.2,1) both}
.res>*:nth-child(2){animation-delay:.1s}.res>*:nth-child(3){animation-delay:.2s}
.hero-result{position:relative;overflow:hidden;padding:34px;border-radius:32px;color:#fff;
  background:linear-gradient(130deg,#7c3aed,#4f46e5 45%,#0ea5e9);box-shadow:0 30px 60px -26px rgba(79,70,229,.8)}
.k{font-size:14px;font-weight:600;margin:0;color:rgba(255,255,255,.8)}
.dz{font-size:clamp(30px,5vw,50px);font-weight:800;letter-spacing:-.03em;margin:6px 0 16px;line-height:1.1;color:#fff!important}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.chips em{font-style:normal;font-size:13px;font-weight:600;padding:6px 13px;border-radius:99px;
  background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.35);color:#fff}
.bars{display:grid;gap:12px}
.bar{display:grid;grid-template-columns:minmax(110px,200px) 1fr 44px;gap:14px;align-items:center;font-size:14px;font-weight:600;color:#fff}
.bar span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar b{text-align:right}
.track{height:10px;border-radius:99px;background:rgba(255,255,255,.25);overflow:hidden}
.track i{display:block;height:100%;width:var(--w);border-radius:99px;background:#fff;animation:grow 1.1s .3s cubic-bezier(.2,.8,.2,1) both}
.bar:not(:first-child) .track i{opacity:.6}
@keyframes grow{from{width:0}}
.note{padding:15px 20px;border-radius:18px;color:#9a3412;font-weight:500;font-size:14px;background:#ffedd5;border:1px solid #fed7aa}
.ins-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.ins{padding:24px 26px;border-radius:28px}
.ins:nth-child(1){background:#f1e9ff}.ins:nth-child(2){background:#dcf7f3}
.ins:nth-child(3){background:#ffe9df}.ins:nth-child(4){background:#e0efff}
.ins h3{font-size:19px;font-weight:800;margin:0 0 10px;color:var(--ink)!important}
.ins ul{margin:0;padding-left:20px;color:#3d4468;line-height:1.7;font-size:15px}
.ins li::marker{color:var(--violet)}

/* ---------- loading ---------- */
.loading{padding:34px;text-align:center}
.orb{display:flex;justify-content:center;gap:10px;margin-bottom:12px}
.orb span{width:16px;height:16px;border-radius:50%;background:var(--violet);animation:bounce 1s infinite ease-in-out}
.orb span:nth-child(2){animation-delay:.15s;background:var(--sky)}.orb span:nth-child(3){animation-delay:.3s;background:var(--teal)}
@keyframes bounce{0%,80%,100%{transform:scale(.5);opacity:.4}40%{transform:translateY(-10px) scale(1);opacity:1}}
.load-text{color:var(--muted);font-weight:600;margin:0 0 20px}
.skel{height:14px;border-radius:8px;margin:10px auto;max-width:560px;
  background:linear-gradient(90deg,#ece6fa,#fff,#ece6fa);background-size:200% 100%;animation:shimmer 1.4s infinite linear}
.skel.short{max-width:340px}
@keyframes shimmer{to{background-position:-200% 0}}

/* ---------- chat ---------- */
.gradio-container .message{border-radius:20px!important;border:0!important;animation:rise .35s both;box-shadow:0 8px 20px -12px rgba(29,27,63,.3)}
.gradio-container .message.user,.gradio-container .user .message{background:linear-gradient(120deg,#7c3aed,#0ea5e9)!important}
.gradio-container .message.user *,.gradio-container .user .message *{color:#fff!important}
.gradio-container .message.bot,.gradio-container .bot .message{background:#fff!important;color:var(--ink)!important}
.gradio-container button.submit-button{background:linear-gradient(120deg,#7c3aed,#14b8a6)!important;color:#fff!important;border-radius:99px!important}
.gradio-container .examples button,.gradio-container .example{border-radius:99px!important;background:#fff!important;
  transition:transform .25s cubic-bezier(.3,1.6,.5,1)!important}
.gradio-container .examples button:hover{transform:translateY(-3px) rotate(-1deg)}

@media (max-width:700px){.hero{padding:40px 24px 80px}.sticker{display:none}.bar{grid-template-columns:90px 1fr 40px}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
:focus-visible{outline:3px solid rgba(124,58,237,.5)!important;outline-offset:2px}
"""

# ==========================================
# UI
# ==========================================
with gr.Blocks(css=css, head=head, title="My AMIGO") as app:
    gr.HTML(HERO)

    with gr.Tabs():
        with gr.Tab("🔬 Symptom checker"):
            with gr.Column(elem_classes="panel spot"):
                gr.HTML('<h2 class="panel-title">What’s bothering you today?</h2>'
                        '<p class="panel-sub">Tap a common symptom or search for your own.</p>')
                if quick:
                    with gr.Row(elem_classes="pills"):
                        quick_btns = [gr.Button(f"+ {pretty(s)}", elem_classes="pill", size="sm") for s in quick]
                symptom_selector = gr.Dropdown(
                    choices=choices, multiselect=True, label="Your symptoms",
                    info="Type to search all symptoms", filterable=True,
                )
                with gr.Row():
                    predict_btn = gr.Button("✨ Analyze symptoms", elem_classes="magnet", scale=3)
                    clear_btn = gr.Button("Clear", elem_classes="ghost", scale=1)

            output = gr.HTML()

            for btn, s in zip(quick_btns if quick else [], quick):
                btn.click(lambda cur, s=s: (cur or []) if s in (cur or []) else (cur or []) + [s],
                          symptom_selector, symptom_selector, show_progress="hidden")
            predict_btn.click(predict_disease, symptom_selector, output, show_progress="hidden")
            clear_btn.click(lambda: ([], ""), None, [symptom_selector, output])

        with gr.Tab("💬 Chat with AMIGO"):
            gr.ChatInterface(
                fn=chat_with_amigo,
                chatbot=gr.Chatbot(height=460, show_label=False),
                textbox=gr.Textbox(placeholder="Ask anything about how you feel…", container=False, scale=7),
                examples=[
                    "I’ve had a headache since yesterday",
                    "How can I sleep better?",
                    "What should I eat when I have a cold?",
                ],
                cache_examples=False,
            )

if __name__ == "__main__":
    app.launch()
