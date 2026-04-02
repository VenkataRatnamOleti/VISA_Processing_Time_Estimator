const API_BASE = window.__API_BASE || "http://127.0.0.1:5000";

const GEMINI_API_KEY = "AIzaSyBrx95Yn7JJXF-sLX8wjm5mmg5UegWs9o0"; // ⚠️ Replace with your key (not secure for prod)

async function generateGeminiResponse(prompt, model = "gemini-2.0-flash") {
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text: prompt,
                },
              ],
            },
          ],
        }),
      }
    );

    const data = await response.json();
    // Access generated text safely
    const output = data?.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated";
    return { success: true, reply: output, model };
  } catch (error) {
    console.error("Gemini Error:", error);
    return { success: false, error: error.message };
  }
}


async function fetchStats(){
  const res = await fetch(`${API_BASE}/api/stats`);
  return res.json();
}

function setText(id, text){
  const el = document.getElementById(id); if(el) el.textContent = text;
}

async function init(){
  try{
    const s = await fetchStats();
    if(s.success){
      // Populate common fields if present on the page
      setText('model-status', s.stats.model_loaded ? 'Loaded' : 'Offline');
      // also update chat-specific status badge if present
      const chatBadge = document.getElementById('chat-model-status');
      if(chatBadge){
        chatBadge.textContent = s.stats.model_loaded ? 'Loaded' : 'Offline';
        chatBadge.className = 'badge ' + (s.stats.model_loaded ? 'bg-success' : 'bg-secondary');
      }
      setText('model-rmse', s.stats.rmse ?? 'N/A');
      setText('model-features', s.stats.n_features ?? 0);
      setText('stats-rmse', s.stats.rmse ?? 'N/A');
      setText('stats-features', s.stats.n_features ?? 0);
  // model sample metrics
  setText('reg-rmse', s.stats.regression_rmse_sample ?? (s.stats.rmse ?? 'N/A'));
  setText('data-samples', s.stats.data_samples ?? '-');
  setText('clf-accuracy', s.stats.classification_accuracy_sample ?? '-');
  setText('clf-roc', s.stats.classification_roc_auc_sample ?? '-');
      // Populate additional analytics fields when available
      setText('visit-count', s.stats.visit_count ?? (s.stats.visits ? s.stats.visits.reduce((a,b)=>a+b,0) : 0));
      setText('avg-rating', s.stats.avg_rating ?? 'N/A');
      setText('model-accuracy', (s.stats.model_accuracy_estimate != null) ? (s.stats.model_accuracy_estimate + '%') : 'N/A');
  // latest verdict from most recent acceptance prediction
  if(s.stats.latest_verdict) setText('latest-verdict', s.stats.latest_verdict);

    // Populate insights overview fields when present on the Insights page
    setText('avg-processing', s.stats.avg_processing_days ?? (s.stats.synthetic_processing_days && s.stats.synthetic_processing_days.length ? Math.round(s.stats.synthetic_processing_days.reduce((a,b)=>a+b,0)/s.stats.synthetic_processing_days.length) : 'N/A'));
    setText('median-processing', s.stats.median_processing_days ?? 'N/A');
    setText('top-states', (s.stats.top_states && s.stats.top_states.join(', ')) ?? '-');
    setText('top-features', (s.stats.top_features && s.stats.top_features.join(', ')) ?? '-');

      // If the page has daysChart, render distribution chart
      const daysCanvas = document.getElementById('daysChart');
      if(daysCanvas){
        const daysCtx = daysCanvas.getContext('2d');
        new Chart(daysCtx, {
          type: 'bar',
          data: { labels: s.stats.synthetic_processing_days.map((d,i)=>`#${i+1}`), datasets:[{label:'Days', data: s.stats.synthetic_processing_days, backgroundColor:'#0d6efd'}] },
          options:{responsive:true}
        });
      }

      // If the page has visaChart, render breakdown chart
      const visaCanvas = document.getElementById('visaChart');
      if(visaCanvas){
        const visaCtx = visaCanvas.getContext('2d');
        new Chart(visaCtx, {
          type: 'doughnut',
          data: { labels: Object.keys(s.stats.visa_type_breakdown), datasets:[{data:Object.values(s.stats.visa_type_breakdown), backgroundColor:['#0d6efd','#6c757d','#198754','#fd7e14'] }] },
          options:{responsive:true}
        });
      }

      // visitChart (monthly) if present on page
      const visitCanvas = document.getElementById('visitChart');
      if(visitCanvas){
        const visitCtx = visitCanvas.getContext('2d');
        const visits = s.stats.visits || [120,150,180,200,250];
        new Chart(visitCtx, {
          type: 'bar',
          data: { labels: visits.map((_,i)=>['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i] || `#${i+1}`), datasets:[{label:'Visits', data: visits, backgroundColor:'rgba(54,162,235,0.6)'}] },
          options:{responsive:true}
        });
      }

      // feedbackChart distribution if present
      const feedbackCanvas = document.getElementById('feedbackChart');
      if(feedbackCanvas){
        const feedbackCtx = feedbackCanvas.getContext('2d');
        const fb = s.stats.feedback_distribution || {positive:70, neutral:20, negative:10};
        new Chart(feedbackCtx, {
          type: 'pie',
          data: { labels: Object.keys(fb).map(k=>k.charAt(0).toUpperCase()+k.slice(1)), datasets:[{data: Object.values(fb), backgroundColor:['rgba(75,192,192,0.6)','rgba(255,206,86,0.6)','rgba(255,99,132,0.6)']}] },
          options:{responsive:true}
        });
      }
    }
  }catch(e){
    console.error('Could not load stats', e);
  }
}

async function submitPredict(ev){
  // Deprecated single-flow handler — replaced by separate estimate and acceptance handlers.
}

async function submitEstimate(ev){
  // Be defensive: handler may be called with an Event (normal) or bound/called from the form
  // (e.g. some older inline onsubmit usage). Try to prevent default if possible.
  try{ if(ev && typeof ev.preventDefault === 'function') ev.preventDefault(); }
  catch(e){}

  // If the handler was called as a method on the form (this === form) or without an event,
  // prefer using the form element as the base for lookups to avoid ambiguous ids.
  const baseForm = (ev && ev.target && ev.target.tagName === 'FORM') ? ev.target : (this && this.tagName === 'FORM' ? this : document.getElementById('estimate-form'));

  // helper to safely read values and avoid null.value errors
  const safeVal = (id, opts = {}) => {
    // prefer query within the form if available, otherwise global lookup
    const el = baseForm ? baseForm.querySelector(`#${id}`) || document.getElementById(id) : document.getElementById(id);
    if(!el) {
      console.warn(`Missing element for id=${id}`);
      return opts.default ?? null;
    }
    const v = el.value;
    if(opts.number){
      if(v === null || v === undefined || v === '') return opts.default ?? null;
      const n = Number(v);
      return Number.isFinite(n) ? n : (opts.default ?? null);
    }
    return v;
  };

  const payload = {
    visa_type: safeVal('visa_type'),
    processing_center: safeVal('processing_center'),
    education_of_employee: safeVal('education_of_employee'),
    has_job_experience: safeVal('has_job_experience'),
    requires_job_training: safeVal('requires_job_training'),
    job_offer: safeVal('job_offer'),
    documents_complete: safeVal('documents_complete'),
    years_experience: safeVal('years_experience', {number:true, default: null}),
    previous_visa_rejections: safeVal('previous_visa_rejections', {number:true, default: 0}),
    application_date: safeVal('application_date', {default: null})
  };

  try{
    const res = await fetch(`${API_BASE}/api/estimate-days`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!res.ok){
      const txt = await res.text().catch(()=>'<no body>');
      throw new Error(`estimate-days failed: ${res.status} ${res.statusText} - ${txt}`);
    }
    const js = await res.json();
    if(!js.success) throw new Error(js.error || 'estimate-days returned failure');
    // Animate the displayed estimate with a small loader then number animation
    const estEl = document.getElementById('est-days');
    const windowEl = document.getElementById('est-window');
    if(estEl){
      estEl.textContent = '';
      const loader = document.createElement('span'); loader.className = 'result-loading'; estEl.appendChild(loader);
      setTimeout(()=>{
        try{ if(loader.parentNode) loader.parentNode.removeChild(loader); }catch(e){}
        animateNumber(estEl, Number(js.estimated_days || 0), 900);
      }, 240);
    }
    if(windowEl) windowEl.textContent = js.window ?? '—';
    // Optionally populate acceptance input
    const acceptInput = document.getElementById('accept_estimated_days');
    if(acceptInput && (acceptInput.value === '' || acceptInput.value === null)){
      acceptInput.value = js.estimated_days ?? '';
    }
  }catch(e){
    console.error('Error in submitEstimate:', e);
    alert('Could not reach API: ' + (e.message || String(e)));
  }
}

async function submitAcceptance(ev){
  ev.preventDefault();
  const safeVal = (id, opts = {}) => {
    const el = document.getElementById(id);
    if(!el) { console.warn(`Missing element for id=${id}`); return opts.default ?? null; }
    const v = el.value;
    if(opts.number){ if(v === null || v === undefined || v === '') return opts.default ?? null; const n = Number(v); return Number.isFinite(n)? n : (opts.default ?? null); }
    return v;
  };
  const days = safeVal('accept_estimated_days', {number:true, default: 0});
  // Build payload using the same form values as estimation so the classifier has full context
  const payload = {
    visa_type: safeVal('visa_type'),
    processing_center: safeVal('processing_center'),
    education_of_employee: safeVal('education_of_employee'),
    has_job_experience: safeVal('has_job_experience'),
    requires_job_training: safeVal('requires_job_training'),
    job_offer: safeVal('job_offer'),
    documents_complete: safeVal('documents_complete'),
    years_experience: safeVal('years_experience', {number:true, default: null}),
    previous_visa_rejections: safeVal('previous_visa_rejections', {number:true, default: 0}),
    application_date: safeVal('application_date', {default: null}),
    estimated_days: days
  };
  try{
    const res = await fetch(`${API_BASE}/api/estimate-acceptance`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!res.ok){
      const txt = await res.text().catch(()=>'<no body>');
      throw new Error(`estimate-acceptance failed: ${res.status} ${res.statusText} - ${txt}`);
    }
    const js = await res.json();
    if(js.success){
      const verdictEl = document.getElementById('est-verdict');
  // API may return 'Accepted'/'Rejected' or 'Approved'/'Rejected' — normalize
  let verdict = js.verdict ?? js.label ?? '—';
  if(verdict === 'Approved') verdict = 'Accepted';
  verdictEl.textContent = verdict;
  verdictEl.className = 'badge ' + (verdict === 'Accepted' ? 'bg-success' : 'bg-danger');
  setText('est-score', `(${js.acceptance_score ?? js.acceptance_score ?? js.acceptance_score ?? js.probability_approved ?? '-'})`);
      // brief pulse to emphasize acceptance result
      try{ verdictEl.classList.add('pulse-accept'); setTimeout(()=>verdictEl.classList.remove('pulse-accept'), 700); }catch(e){}
        // After a successful prediction, request LLM-powered insights (Gemini) and surface them under the Predict page
        try{ const summary = `Estimated days: ${js.estimated_days ?? days}; Verdict: ${verdict}; Acceptance score: ${js.acceptance_score ?? '-'}.`; generateInsights(summary); }catch(e){}
      } else {
      console.warn('acceptance call failed', js.error);
      setText('est-verdict', '—');
      setText('est-score', '-');
    }
  }catch(e){
    console.error('Error in submitAcceptance:', e);
    alert('Could not reach API: ' + (e.message || String(e)));
  }
}

function clearForm(){
  const ids = ['visa_type','processing_center','education_of_employee','has_job_experience','requires_job_training','job_offer','documents_complete','years_experience','previous_visa_rejections','application_date','accept_estimated_days'];
  ids.forEach(id=>{ const el=document.getElementById(id); if(!el) return; if(el.tagName==='SELECT') el.selectedIndex = 0; else el.value = ''; });
}

// simple number animator (writes to element's textContent)
function animateNumber(el, target, duration = 800){
  if(!el) return;
  const start = 0;
  const startTime = performance.now();
  el.classList.add('animating');
  function step(now){
    const t = Math.min((now - startTime) / duration, 1);
    const eased = 1 - (1 - t) * (1 - t); // easeOutQuad
    const value = Math.round(start + (target - start) * eased);
    el.textContent = value;
    if(t < 1) requestAnimationFrame(step);
    else el.classList.remove('animating');
  }
  requestAnimationFrame(step);
}

async function sendChat(){
  const input = document.getElementById('chat-input');
  const text = input.value.trim(); if(!text) return;
  // append user message and show a typing bubble while waiting
  appendChat('user', text);
  input.value = '';
  // create typing indicator bubble
  const win = document.getElementById('chat-window');
  const typingId = 'bot-typing';
  const typingDiv = document.createElement('div'); typingDiv.id = typingId; typingDiv.className = 'chat-msg bot';
  const typingBubble = document.createElement('div'); typingBubble.className = 'bubble muted'; typingBubble.textContent = '…';
  typingDiv.appendChild(typingBubble); win.appendChild(typingDiv); win.scrollTop = win.scrollHeight;

  try{
    const prompt = `User: ${text}\nAssistant:`;
    const result = await generateGeminiResponse(prompt);
    // remove typing indicator
    try{ const t = document.getElementById(typingId); if(t) t.remove(); }catch(e){}
    if(result.success){ 
      appendChat('bot', result.reply, result.model); 
    } else {
      appendChat('bot', `Error: ${result.error || 'Gemini unavailable'}`);
    }
  }catch(e){
    try{ const t = document.getElementById(typingId); if(t) t.remove(); }catch(er){}
    appendChat('bot', 'Chat unavailable - check console');
  }
}

function appendChat(sender, text, meta){
  const win = document.getElementById('chat-window');
  const div = document.createElement('div'); div.className = 'chat-msg ' + (sender==='user'?'user':'bot');
  const bubble = document.createElement('div'); bubble.className = 'bubble';
  // Render bot replies with simple formatting: escape HTML then convert newlines to <br>
  const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  if(sender === 'bot'){
    bubble.innerHTML = escapeHtml(text || '').replace(/\n\n/g, '<p></p>').replace(/\n/g, '<br>');
  } else {
    bubble.textContent = text;
  }
  div.appendChild(bubble);
  // meta (source/model) shown as small muted line beneath bubble
  if(meta){
    const m = document.createElement('div'); m.className = 'small muted mt-1'; m.style.marginTop='6px'; m.textContent = String(meta).replace(/([A-Z])/g, ' $1').trim(); div.appendChild(m);
  }
  win.appendChild(div); win.scrollTop = win.scrollHeight;
}

let isInsightsLoading = false;

async function generateInsights(text){
  if(isInsightsLoading) return;
  isInsightsLoading = true;

  const outCard = document.getElementById('insights-card');
  const out = document.getElementById('insights-output');
  const err = document.getElementById('insights-error');

  try{
    if(!outCard || !out) return;

    outCard.style.display = 'block';

    if(err){
      err.style.display = 'none';
      err.textContent = '';
    }

    // Loading UI
    out.innerHTML = `<span class="text-muted">⏳ Generating insights...</span>`;
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text: `Provide concise customer-service suggestions for:\n\n${text}`
                }
              ]
            }
          ]
        })
      }
    );

    if(!res.ok){
      const errText = await res.text();
      throw new Error(`Gemini ${res.status}: ${errText}`);
    }

    const data = await res.json();

    const reply =
      data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "No response generated";

    // Format output nicely
    const html = reply
      .split('\n\n')
      .map(p => `<p>${p.replace(/</g,'&lt;').replace(/\n/g,'<br>')}</p>`)
      .join('');

    out.innerHTML = html;

  } catch(e){
    console.error("Gemini Error:", e);

    if(out) out.innerHTML = '';

    if(err){
      err.style.display = 'block';

      if(String(e.message).includes('429')){
        err.textContent = "⚠️ Too many requests. Please wait...";
      }
      else if(String(e.message).includes('404')){
        err.textContent = "⚠️ Model not found. Check API.";
      }
      else{
        err.textContent = "Error: " + (e.message || e);
      }
    }

  } finally {
    isInsightsLoading = false;
  }
}
// Fire-and-forget: track a visit for analytics (frontend call increments server-side counter)
async function trackVisit(){
  try{ await fetch(`${API_BASE}/api/track-visit`, {method:'POST'}).catch(()=>{}); }
  catch(e){}
}

document.addEventListener('DOMContentLoaded', ()=>{
  init();
  // track a site visit (non-blocking)
  try{ trackVisit(); }catch(e){}

  // Theme initialization: central handler for all pages
  function applyTheme(theme){
    if(theme === 'dark') document.documentElement.setAttribute('data-theme','dark');
    else document.documentElement.removeAttribute('data-theme');
  }
  function currentTheme(){ return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'; }
  // initialize from localStorage
  try{
    const stored = localStorage.getItem('theme'); if(stored) applyTheme(stored);
  }catch(e){}
  const themeBtn = document.getElementById('themeToggle');
  if(themeBtn){
    const setThemeButton = (theme)=>{
      themeBtn.setAttribute('data-theme', theme);
      if(theme === 'dark'){
        themeBtn.classList.add('toggled');
        themeBtn.textContent = '🌙';
      } else {
        themeBtn.classList.remove('toggled');
        themeBtn.textContent = '☀️';
      }
    };
    themeBtn.addEventListener('click', ()=>{
      const next = currentTheme() === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      try{ localStorage.setItem('theme', next); }catch(e){}
      setThemeButton(next);
    });
    // ensure button reflects initial state
    setThemeButton(currentTheme());
  }

  // Sidebar collapse feature removed — sidebar remains static.

  // Attach predict form handlers only if the form exists on this page
  // Predict page: estimate and acceptance forms
  const estimateForm = document.getElementById('estimate-form');
  if(estimateForm){ estimateForm.addEventListener('submit', submitEstimate); }
  const acceptanceForm = document.getElementById('acceptance-form');
  if(acceptanceForm){ acceptanceForm.addEventListener('submit', submitAcceptance); }
  const useEstimateBtn = document.getElementById('use-estimate');
  if(useEstimateBtn){ useEstimateBtn.addEventListener('click', ()=>{
    const days = document.getElementById('est-days').textContent;
    const input = document.getElementById('accept_estimated_days');
    if(input && days && days !== '—') input.value = days;
  }); }
  const clearBtn = document.getElementById('clear-btn');
  if(clearBtn){ clearBtn.addEventListener('click', clearForm); }

  // Rating buttons on dashboard
  const ratingButtons = document.getElementById('rating-buttons');
  if(ratingButtons){
    ratingButtons.addEventListener('click', async (e)=>{
      const btn = e.target.closest('button[data-rate]'); if(!btn) return;
      const r = Number(btn.getAttribute('data-rate')) || 0;
      try{
        const res = await fetch(`${API_BASE}/api/rate`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rating: r})});
        const js = await res.json();
        if(js.success){ setText('avg-rating', js.avg_rating ?? 'N/A'); alert('Thanks for rating!'); }
        else alert('Rating failed: ' + (js.error||'unknown'));
      }catch(e){ alert('Could not send rating: ' + e.message); }
    });
  }

  // Chat handlers only on chat page
  const chatSend = document.getElementById('chat-send');
  const chatInput = document.getElementById('chat-input');
  if(chatSend){ chatSend.addEventListener('click', sendChat); }
  if(chatInput){ chatInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendChat(); }); }

  // Clear and copy controls (chat page)
  const chatClear = document.getElementById('chat-clear');
  if(chatClear){ chatClear.addEventListener('click', ()=>{ const win = document.getElementById('chat-window'); if(win) win.innerHTML=''; }); }
  const chatCopy = document.getElementById('chat-copy');
  if(chatCopy){ chatCopy.addEventListener('click', async ()=>{ const win = document.getElementById('chat-window'); if(!win) return; try{ const text = Array.from(win.querySelectorAll('.chat-msg')).map(n=>n.innerText.trim()).join('\n\n'); await navigator.clipboard.writeText(text); alert('Chat copied to clipboard'); }catch(e){ alert('Copy failed: '+(e.message||e)); } }); }
  // Insights copy/save handlers on Predict page
  const insightsCopy = document.getElementById('copy-insights');
  if(insightsCopy){ insightsCopy.addEventListener('click', async ()=>{ const out = document.getElementById('insights-output'); if(!out) return; try{ const text = out.innerText || out.textContent || ''; if(!text) { alert('No insights to copy'); return; } await navigator.clipboard.writeText(text); alert('Insights copied to clipboard'); }catch(e){ alert('Copy failed: '+(e.message||e)); } }); }
  const insightsSave = document.getElementById('save-pdf');
  if(insightsSave){ insightsSave.addEventListener('click', ()=>{ const out = document.getElementById('insights-output'); const text = out ? out.innerText.trim() : ''; if(!text){ alert('No insights to save'); return; } try{ const { jsPDF } = window.jspdf || {}; if(!jsPDF){ alert('PDF export not available'); return; } const doc = new jsPDF(); const lines = doc.splitTextToSize(text, 180); doc.setFontSize(12); doc.text('Customer Suggestions', 10, 15); doc.setFontSize(10); doc.text(lines, 10, 25); doc.save('insights.pdf'); }catch(e){ alert('Save PDF failed: '+(e.message||e)); } }); }

  // Outputs preview removed — chat and insights now use full content areas.
});
