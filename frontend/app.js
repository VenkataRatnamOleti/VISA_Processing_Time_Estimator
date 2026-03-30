// Use environment variable for API_BASE or fallback to localhost
const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:5000/api';

async function fetchStats(){
  const res = await fetch(`${API_BASE}/stats`);
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
      setText('model-rmse', s.stats.rmse ?? 'N/A');
      setText('model-features', s.stats.n_features ?? 0);
      setText('stats-rmse', s.stats.rmse ?? 'N/A');
      setText('stats-features', s.stats.n_features ?? 0);

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
    }
  }catch(e){
    console.error('Could not load stats', e);
  }
}

async function submitPredict(ev){
  ev.preventDefault();
  const payload = {
    prevailing_wage: Number(document.getElementById('prevailing_wage').value || 0),
    unit_of_wage: document.getElementById('unit_of_wage').value,
    yr_of_estab: Number(document.getElementById('yr_of_estab').value || 2000),
    no_of_employees: Number(document.getElementById('no_of_employees').value || 0),
    visa_type: document.getElementById('visa_type').value
  };

  try{
    // 1) Request estimated days
    const resDays = await fetch(`${API_BASE}/estimate-days`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!resDays.ok){
      const txt = await resDays.text().catch(()=>'<no body>');
      throw new Error(`estimate-days failed: ${resDays.status} ${resDays.statusText} - ${txt}`);
    }
  const jsDays = await resDays.json();
  if(!jsDays.success){ throw new Error(jsDays.error || 'estimate-days reported failure'); }
  // Backend returns estimated_days and window at the top level
  setText('est-days', jsDays.estimated_days ?? '—');
  setText('est-window', jsDays.window ?? '—');

    // 2) Request acceptance separately (can use estimated_days only)
  const accPayload = { estimated_days: jsDays.estimated_days };
    const resAcc = await fetch(`${API_BASE}/estimate-acceptance`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(accPayload)});
    if(!resAcc.ok){
      const txt = await resAcc.text().catch(()=>'<no body>');
      throw new Error(`estimate-acceptance failed: ${resAcc.status} ${resAcc.statusText} - ${txt}`);
    }
    const jsAcc = await resAcc.json();
    if(jsAcc.success){
      const verdictEl = document.getElementById('est-verdict');
      verdictEl.textContent = jsAcc.verdict ?? '—';
      verdictEl.className = 'badge ' + (jsAcc.verdict === 'Accepted' ? 'bg-success' : 'bg-danger');
      setText('est-score', `(${jsAcc.acceptance_score ?? '-'})`);
      setText('latest-verdict', jsAcc.verdict ?? '—');
    } else {
      // If acceptance fails, show placeholder and log
      console.warn('acceptance call failed', jsAcc.error);
      setText('est-verdict', '—');
      setText('est-score', '-');
    }
  }catch(e){
    console.error(e);
    // Surface the actual error message to help debugging (status codes, body, etc.)
    alert('Could not reach API: ' + (e.message || String(e)));
  }
}

function clearForm(){
  document.getElementById('prevailing_wage').value = '';
  document.getElementById('yr_of_estab').value = '';
  document.getElementById('no_of_employees').value = '';
}

async function sendChat(){
  const input = document.getElementById('chat-input');
  const text = input.value.trim(); if(!text) return;
  appendChat('user', text);
  input.value = '';
  try{
    const res = await fetch(`${API_BASE}/chat`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:text})});
    const js = await res.json();
    if(js.success){ appendChat('bot', js.reply); }
    else appendChat('bot', 'Error: ' + (js.error||'unknown'));
  }catch(e){ appendChat('bot', 'Chat service unavailable'); }
}

function appendChat(sender, text){
  const win = document.getElementById('chat-window');
  const div = document.createElement('div'); div.className = 'chat-msg ' + (sender==='user'?'user':'bot');
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
  div.appendChild(bubble); win.appendChild(div); win.scrollTop = win.scrollHeight;
}

document.addEventListener('DOMContentLoaded', ()=>{
  init();

  // Attach predict form handlers only if the form exists on this page
  const predictForm = document.getElementById('predict-form');
  if(predictForm){
    predictForm.addEventListener('submit', submitPredict);
  }
  const clearBtn = document.getElementById('clear-btn');
  if(clearBtn){ clearBtn.addEventListener('click', clearForm); }

  // Chat handlers only on chat page
  const chatSend = document.getElementById('chat-send');
  const chatInput = document.getElementById('chat-input');
  if(chatSend){ chatSend.addEventListener('click', sendChat); }
  if(chatInput){ chatInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendChat(); }); }
});
