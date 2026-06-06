/* ───────── MimicAI v2 — Frontend ───────── */

const API = {
  chat: '/api/chat',
  voices: '/api/voices',
  voiceRecord: '/api/voices/record',
  voiceDelete: (id) => `/api/voices/${id}/delete`,
  voicePreview: (id) => `/api/voices/${id}/preview`,
  tts: '/api/tts',
  ttsAsync: '/api/tts/async',
  ttsAsyncStatus: (id) => `/api/tts/async/${id}/status`,
  ttsAsyncAudio: (id) => `/api/tts/async/${id}/audio`,
  languages: '/api/languages',
  status: '/api/status',
};

const KEYS = {
  voice: 'mimicai:voice',
  autoSpeak: 'mimicai:auto-speak',
  lightweightTts: 'mimicai:lightweight-tts',
  theme: 'mimicai:theme',
};

/* ───────── State ───────── */
const state = {
  messages: [],
  voices: [],
  selectedVoice: null,
  isGenerating: false,
  mediaRecorder: null,
  recordedChunks: [],
  recordingTimer: null,
  recordingSeconds: 0,
  autoSpeak: false,
  lightweightTts: false,
  currentAudio: null,
  theme: 'dark',
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

/* ───────── Init ───────── */
document.addEventListener('DOMContentLoaded', () => {
  state.selectedVoice = localStorage.getItem(KEYS.voice) || null;
  state.autoSpeak = localStorage.getItem(KEYS.autoSpeak) === 'true';
  state.lightweightTts = localStorage.getItem(KEYS.lightweightTts) === 'true';
  state.theme = localStorage.getItem(KEYS.theme) || 'dark';
  document.documentElement.setAttribute('data-theme', state.theme);
  bindEvents();
  loadVoices();
  loadLanguages();
  checkEngineStatus();
  // Poll status every 30 seconds
  setInterval(checkEngineStatus, 30000);
});

function bindEvents() {
  $('#send-btn').addEventListener('click', sendMessage);
  $('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $('#chat-input').addEventListener('input', () => {
    const t = $('#chat-input');
    t.style.height = 'auto';
    t.style.height = Math.min(t.scrollHeight, 150) + 'px';
  });

  $('#toggle-sidebar').addEventListener('click', () => $('.sidebar').classList.toggle('hidden'));
  const nb = $('#new-chat-btn'); if (nb) nb.addEventListener('click', newChat);

  $('#record-voice-btn').addEventListener('click', openModal);
  $('#modal-close-btn').addEventListener('click', closeModal);
  $('#modal-cancel-btn').addEventListener('click', closeModal);
  $('#record-btn').addEventListener('click', toggleRecording);
  $('#save-voice-btn').addEventListener('click', saveVoice);

  $('#voice-select').addEventListener('change', (e) => {
    state.selectedVoice = e.target.value || null;
    persistVoice();
    renderVoiceList();
  });

  const as = $('#autospeak-toggle');
  as.checked = state.autoSpeak;
  as.addEventListener('change', (e) => {
    state.autoSpeak = e.target.checked;
    localStorage.setItem(KEYS.autoSpeak, String(state.autoSpeak));
  });

  const lt = $('#lightweight-toggle');
  if (lt) {
    lt.checked = state.lightweightTts;
    lt.addEventListener('change', (e) => {
      state.lightweightTts = e.target.checked;
      localStorage.setItem(KEYS.lightweightTts, String(state.lightweightTts));
      toast(state.lightweightTts ? 'Lightweight OmniVoice enabled' : 'Full OmniVoice enabled', 'info');
    });
  }

  $$('.welcome-pill').forEach(el => {
    el.addEventListener('click', () => {
      $('#chat-input').value = el.dataset.prompt || el.textContent.trim();
      $('#chat-input').focus();
    });
  });

  const sab = $('#stop-audio-btn');
  if (sab) {
    sab.addEventListener('click', () => {
      if (state.currentAudio) {
        state.currentAudio.pause();
        setCurrentAudio(null);
      }
    });
  }

  const themeToggle = $('#theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      state.theme = state.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', state.theme);
      localStorage.setItem(KEYS.theme, state.theme);
    });
  }

  $$('.symbol-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sym = btn.dataset.symbol;
      const ta = $('#chat-input');
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const text = ta.value;
      ta.value = text.substring(0, start) + sym + text.substring(end);
      ta.focus();
      ta.selectionStart = ta.selectionEnd = start + sym.length;
      ta.dispatchEvent(new Event('input'));
    });
  });
}

/* ───────── Chat ───────── */
async function sendMessage() {
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text || state.isGenerating) return;

  addMsg('user', text);
  input.value = ''; input.style.height = 'auto';
  hideWelcome();

  state.isGenerating = true;
  setSendBtn(true);
  showTyping();

  try {
    const res = await fetch(API.chat, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: state.messages.slice(-10) }),
    });

    removeTyping();

    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      addMsg('ai', `Error: ${e.error || 'Unknown error'}`);
      return;
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let aiText = '', buf = '';
    const msgEl = addMsg('ai', '', true);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        const t = line.trim();
        if (!t || !t.startsWith('data: ')) continue;
        const d = t.slice(6);
        if (d === '[DONE]') continue;
        try {
          const c = JSON.parse(d).choices?.[0]?.delta?.content;
          if (c) { aiText += c; updateBubble(msgEl, aiText); }
        } catch {}
      }
    }
    // flush remaining buffer
    if (buf.trim().startsWith('data: ')) {
      const d = buf.trim().slice(6);
      if (d !== '[DONE]') {
        try {
          const c = JSON.parse(d).choices?.[0]?.delta?.content;
          if (c) { aiText += c; updateBubble(msgEl, aiText); }
        } catch {}
      }
    }

    if (aiText.trim()) {
      state.messages.push({ role: 'assistant', content: aiText });
      addActions(msgEl, aiText);
      // Auto-play TTS if desired
      if (state.autoSpeak && state.selectedVoice) {
        const btn = msgEl.querySelector('.tts-btn');
        if (btn) playTts(aiText, btn);
      }
    } else {
      msgEl.remove();
      addMsg('ai', 'Sorry, no response was generated. Please try again.');
    }
  } catch (err) {
    removeTyping();
    addMsg('ai', `Connection error: ${err.message}`);
  } finally {
    state.isGenerating = false;
    setSendBtn(false);
  }
}

function addMsg(role, text, streaming = false) {
  const c = $('#messages');
  const isAi = role === 'ai' || role === 'assistant';
  const el = document.createElement('div');
  el.className = `message ${isAi ? 'ai' : 'user'}`;
  el.innerHTML = `
    <div class="message-avatar">${isAi ? '✦' : '👤'}</div>
    <div class="message-content">
      <div class="message-bubble">${isAi ? md(text) : esc(text)}</div>
      <div class="message-actions"></div>
    </div>`;
  c.appendChild(el);
  scroll();
  if (!streaming && text) {
    state.messages.push({ role: role === 'user' ? 'user' : 'assistant', content: text });
  }
  return el;
}

function updateBubble(el, text) {
  const b = el.querySelector('.message-bubble');
  if (b) { b.innerHTML = md(text); scroll(); }
}

function addActions(el, text) {
  const a = el.querySelector('.message-actions');
  if (!a) return;

  // Listen button
  const tb = document.createElement('button');
  tb.className = 'msg-action-btn tts-btn';
  tb.innerHTML = '🔊 Listen';
  tb.onclick = () => playTts(text, tb);
  a.appendChild(tb);

  // Copy button
  const cb = document.createElement('button');
  cb.className = 'msg-action-btn';
  cb.innerHTML = '📋 Copy';
  cb.onclick = () => {
    navigator.clipboard.writeText(text).then(() => {
      cb.innerHTML = '✓ Copied';
      setTimeout(() => { cb.innerHTML = '📋 Copy'; }, 2000);
    });
  };
  a.appendChild(cb);
}

function showTyping() {
  const c = $('#messages');
  const el = document.createElement('div');
  el.className = 'typing-indicator'; el.id = 'typing-indicator';
  el.innerHTML = `
    <div class="message-avatar" style="background:linear-gradient(135deg,var(--accent),#b06ef9);color:white">✦</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>`;
  c.appendChild(el); scroll();
}
function removeTyping() { const e = $('#typing-indicator'); if (e) e.remove(); }

function hideWelcome() { const w = $('#welcome-screen'); if (w) w.style.display = 'none'; }

function newChat() {
  state.messages = [];
  const c = $('#messages');
  c.innerHTML = `
    <div class="welcome" id="welcome-screen">
      <div class="welcome-glow"></div>
      <div class="welcome-icon">✦</div>
      <h1>MimicAI</h1>
      <p>Chat with AI and hear responses spoken aloud. Record a voice sample in the sidebar, then start chatting.</p>
      <div class="welcome-pills">
        <div class="welcome-pill" data-prompt="Tell me a joke">💬 Tell me a joke</div>
        <div class="welcome-pill" data-prompt="Explain quantum physics simply">⚛️ Explain quantum physics</div>
        <div class="welcome-pill" data-prompt="Write a short poem about the ocean">🌊 Write a poem</div>
        <div class="welcome-pill" data-prompt="What can you do?">✨ What can you do?</div>
      </div>
    </div>`;
  $$('.welcome-pill').forEach(el => {
    el.addEventListener('click', () => {
      $('#chat-input').value = el.dataset.prompt || el.textContent.trim();
      $('#chat-input').focus();
    });
  });
  toast('New chat started', 'success');
}

function setSendBtn(loading) {
  const b = $('#send-btn');
  b.disabled = loading;
  if (loading) b.innerHTML = '<div class="tts-spinner" style="border-top-color:white"></div>';
  else b.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
}

function scroll() {
  const a = $('#chat-area');
  requestAnimationFrame(() => { a.scrollTop = a.scrollHeight; });
}

/* ───────── TTS ───────── */
async function playTts(text, btn) {
  if (!text || !btn) return;
  const voiceId = state.selectedVoice || null;
  if (!voiceId) { toast('Select a voice to hear responses', 'info'); return; }

  const profile = state.voices.find(v => v.id === voiceId);
  const lang = profile?.language || 'en';
  const orig = btn.innerHTML;

  btn.innerHTML = '⏳ Loading…'; btn.disabled = true;

  try {
    const res = await fetch(API.tts, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text.substring(0, 500),
        voice_id: voiceId,
        language: lang,
        prefer_clone: true,
        lightweight: state.lightweightTts,
        strict_clone: false,
      }),
    });

    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.error || 'TTS failed');
    }

    const engine = res.headers.get('X-TTS-Engine');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    // Create a visible audio player and append it below the message bubble
    let playerContainer = btn.parentElement.querySelector('.audio-player-container');
    if (!playerContainer) {
      playerContainer = document.createElement('div');
      playerContainer.className = 'audio-player-container';
      playerContainer.style.marginTop = '10px';
      playerContainer.style.width = '100%';
      btn.parentElement.appendChild(playerContainer);
    }
    
    playerContainer.innerHTML = `<audio controls src="${url}" style="height:32px; width:100%; outline:none; border-radius:16px;"></audio>`;
    const audio = playerContainer.querySelector('audio');
    
    setCurrentAudio(audio);
    
    btn.innerHTML = '✓ Cloned';
    setTimeout(() => { btn.innerHTML = orig; btn.disabled = false; }, 2000);
    
    audio.play().catch(e => { toast('Playback failed: ' + e.message, 'error'); });
    
    if (engine) toast(`Generated via ${engine}`, 'info');
  } catch (err) {
    btn.innerHTML = orig; btn.disabled = false;
    toast(err.message, 'error');
  }
}

/* ───────── Voice Profiles ───────── */
async function loadVoices() {
  try {
    const r = await fetch(API.voices);
    const d = await r.json();
    state.voices = d.voices || [];
    renderVoiceList();
    updateVoiceSelect();
  } catch (e) { console.warn('Voice load error:', e); }
}

function renderVoiceList() {
  const list = $('#voice-list');
  list.innerHTML = '';
  if (state.voices.length === 0) {
    list.innerHTML = '<li class="voice-empty"><div class="voice-empty-icon">🎤</div><div>No voice profiles yet.</div><div style="font-size:11px;margin-top:2px">Record a sample to enable TTS</div></li>';
    return;
  }
  state.voices.forEach(v => {
    const li = document.createElement('li');
    li.className = `voice-item ${state.selectedVoice === v.id ? 'active' : ''}`;
    li.innerHTML = `
      <div class="voice-avatar">${v.name.charAt(0).toUpperCase()}</div>
      <div class="voice-info">
        <div class="voice-name">${esc(v.name)}</div>
        <div class="voice-lang">${v.language || 'en'} · Reference</div>
      </div>
      <div class="voice-actions">
        <button class="voice-act-btn preview" title="Preview">▶</button>
        <button class="voice-act-btn delete" title="Delete">✕</button>
      </div>`;
    li.addEventListener('click', (e) => {
      if (e.target.closest('.voice-act-btn')) return;
      state.selectedVoice = v.id;
      $('#voice-select').value = v.id;
      persistVoice();
      renderVoiceList();
      toast(`Voice: ${v.name}`, 'success');
    });
    li.querySelector('.preview').addEventListener('click', (e) => { e.stopPropagation(); previewVoice(v, e.currentTarget); });
    li.querySelector('.delete').addEventListener('click', (e) => { e.stopPropagation(); deleteVoice(v.id); });
    list.appendChild(li);
  });
}

function updateVoiceSelect() {
  const sel = $('#voice-select');
  while (sel.options.length > 1) sel.remove(1);
  state.voices.forEach(v => {
    const o = document.createElement('option');
    o.value = v.id; o.textContent = `🎤 ${v.name}`;
    sel.appendChild(o);
  });
  if (state.selectedVoice) {
    const ok = state.voices.some(v => v.id === state.selectedVoice);
    if (ok) sel.value = state.selectedVoice;
    else { state.selectedVoice = null; sel.value = ''; persistVoice(); }
  }
}

function setCurrentAudio(audio) {
  if (state.currentAudio) {
    state.currentAudio.pause();
    state.currentAudio.onplay = null;
    state.currentAudio.onpause = null;
    state.currentAudio.onended = null;
  }
  state.currentAudio = audio;
  const stopBtn = document.getElementById('stop-audio-btn');
  if (audio) {
    audio.onplay = () => { if (stopBtn) stopBtn.style.display = 'flex'; };
    audio.onpause = () => { if (stopBtn) stopBtn.style.display = 'none'; };
    audio.onended = () => { if (stopBtn) stopBtn.style.display = 'none'; setCurrentAudio(null); };
    if (stopBtn) stopBtn.style.display = 'flex';
  } else {
    if (stopBtn) stopBtn.style.display = 'none';
  }
}

function persistVoice() {
  if (state.selectedVoice) localStorage.setItem(KEYS.voice, state.selectedVoice);
  else localStorage.removeItem(KEYS.voice);
}

async function previewVoice(voice, btn) {
  const orig = btn.innerHTML; btn.disabled = true; btn.innerHTML = '…';
  try {
    const r = await fetch(API.voicePreview(voice.id), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language: voice.language || 'en' }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || 'Preview failed');
    const url = URL.createObjectURL(await r.blob());
    const a = new Audio(url);
    setCurrentAudio(a);
    await a.play();
    a.onended = () => { setCurrentAudio(null); URL.revokeObjectURL(url); };
  } catch (e) { toast(e.message, 'error'); }
  finally { btn.disabled = false; btn.innerHTML = orig; }
}

async function deleteVoice(id) {
  if (!confirm('Delete this voice profile?')) return;
  try {
    const r = await fetch(API.voiceDelete(id), { method: 'DELETE' });
    if (r.ok) {
      if (state.selectedVoice === id) { state.selectedVoice = null; $('#voice-select').value = ''; persistVoice(); }
      toast('Voice deleted', 'success'); loadVoices();
    }
  } catch { toast('Failed to delete', 'error'); }
}

/* ───────── Recording ───────── */
function openModal() { $('#record-modal').classList.add('active'); resetRecording(); }
function closeModal() { $('#record-modal').classList.remove('active'); stopRecording(); resetRecording(); }

function resetRecording() {
  state.recordedChunks = [];
  state.recordingSeconds = 0;
  $('#record-timer').textContent = '0:00';
  $('#record-status').textContent = 'Click the mic to start recording';
  $('#record-btn').classList.remove('recording');
  $('#record-bars').classList.remove('active');
  $('#audio-preview').innerHTML = '';
  $('#save-voice-btn').disabled = true;
}

async function toggleRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1, sampleRate: 44100 }
    });
    state.recordedChunks = [];
    let mime = 'audio/webm';
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mime = 'audio/webm;codecs=opus';
    else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) mime = 'audio/ogg;codecs=opus';
    else if (MediaRecorder.isTypeSupported('audio/mp4')) mime = 'audio/mp4';

    state.mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    state.mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) state.recordedChunks.push(e.data); };
    state.mediaRecorder.onstop = () => { stream.getTracks().forEach(t => t.stop()); if (state.recordedChunks.length > 0) showPreview(); };
    state.mediaRecorder.onerror = () => { toast('Recording error', 'error'); stopRecording(); };
    state.mediaRecorder.start(500);

    $('#record-btn').classList.add('recording');
    $('#record-bars').classList.add('active');
    $('#record-status').textContent = 'Recording… Click stop when done';

    state.recordingSeconds = 0;
    state.recordingTimer = setInterval(() => {
      state.recordingSeconds++;
      const m = Math.floor(state.recordingSeconds / 60);
      const s = state.recordingSeconds % 60;
      $('#record-timer').textContent = `${m}:${s.toString().padStart(2, '0')}`;
      if (state.recordingSeconds >= 30) { toast('Max 30s reached', 'info'); stopRecording(); }
    }, 1000);
  } catch (err) {
    if (err.name === 'NotAllowedError') toast('Microphone access denied', 'error');
    else if (err.name === 'NotFoundError') toast('No microphone found', 'error');
    else toast(`Mic error: ${err.message}`, 'error');
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state === 'recording') state.mediaRecorder.stop();
  if (state.recordingTimer) { clearInterval(state.recordingTimer); state.recordingTimer = null; }
  $('#record-btn').classList.remove('recording');
  $('#record-bars').classList.remove('active');
  if (state.recordedChunks.length > 0) $('#record-status').textContent = `Recorded ${state.recordingSeconds}s — Ready to save`;
  else $('#record-status').textContent = 'Click the mic to start recording';
}

function showPreview() {
  const blob = new Blob(state.recordedChunks, { type: state.mediaRecorder?.mimeType || 'audio/webm' });
  const url = URL.createObjectURL(blob);
  $('#audio-preview').innerHTML = `<div style="display:flex;align-items:center;gap:8px;padding:6px 0"><audio controls src="${url}" style="flex:1;height:38px"></audio><span style="font-size:11px;color:var(--text3)">${fmtBytes(blob.size)}</span></div>`;
  $('#save-voice-btn').disabled = false;
}

async function saveVoice() {
  const name = $('#voice-name').value.trim();
  const lang = $('#voice-language').value;
  if (!name) { toast('Enter a profile name', 'error'); $('#voice-name').focus(); return; }
  if (state.recordedChunks.length === 0) { toast('Record your voice first', 'error'); return; }

  const mime = state.mediaRecorder?.mimeType || 'audio/webm';
  const blob = new Blob(state.recordedChunks, { type: mime });
  let ext = 'webm';
  if (mime.includes('ogg')) ext = 'ogg';
  else if (mime.includes('mp4')) ext = 'mp4';

  const fd = new FormData();
  fd.append('audio', blob, `recording.${ext}`);
  fd.append('name', name);
  fd.append('language', lang);

  const btn = $('#save-voice-btn');
  btn.disabled = true; btn.textContent = 'Saving…';

  try {
    const r = await fetch(API.voiceRecord, { method: 'POST', body: fd });
    const d = await r.json().catch(() => ({}));
    if (r.ok && d.success) {
      toast(`Voice "${name}" saved!`, 'success');
      closeModal();
      await loadVoices();
      if (d.voice?.id) {
        state.selectedVoice = d.voice.id;
        $('#voice-select').value = d.voice.id;
        persistVoice(); renderVoiceList();
      }
    } else toast(d.error || 'Save failed', 'error');
  } catch (e) { toast('Connection error: ' + e.message, 'error'); }
  finally { btn.disabled = false; btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> Save Voice Profile'; }
}

/* ───────── Languages ───────── */
async function loadLanguages() {
  try {
    const r = await fetch(API.languages);
    const d = await r.json();
    const langs = d.languages || [];
    const sel = $('#voice-language');
    if (!sel) return;
    const existing = new Set(Array.from(sel.options).map(o => o.value));
    langs.forEach(l => { if (!existing.has(l.id)) { const o = document.createElement('option'); o.value = l.id; o.textContent = l.name; sel.appendChild(o); } });
  } catch {}
}

async function checkEngineStatus() {
  try {
    const r = await fetch(API.status);
    if (!r.ok) return;
    const data = await r.json();
    Object.keys(data).forEach(key => {
      const info = data[key];
      const el = document.getElementById(`status-${key.replace('_', '-')}`);
      if (el) {
        el.className = 'status-badge';
        if (info.loaded) el.classList.add('cached');
        else if (info.available) el.classList.add('online');
      }
    });
  } catch {}
}

/* ───────── Toast ───────── */
function toast(msg, type = 'success') {
  const c = $('#toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  const icon = type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ';
  t.innerHTML = `${icon} ${esc(msg)}`;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(20px)'; setTimeout(() => t.remove(), 300); }, 3500);
}

/* ───────── Utilities ───────── */
function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function md(text) {
  if (!text) return '';
  let h = esc(text);
  
  // 1. Parse code blocks
  const codeBlocks = [];
  h = h.replace(/```(\w*)\n?([\s\S]*?)```/g, (match, lang, code) => {
    const placeholder = `__CODE_BLOCK_PLACEHOLDER_${codeBlocks.length}__`;
    codeBlocks.push(`<pre style="background:rgba(255,255,255,0.04);padding:12px;border-radius:8px;overflow-x:auto;font-size:13px;margin:8px 0;border:1px solid var(--border)"><code>${code}</code></pre>`);
    return placeholder;
  });

  // 2. Parse inline code
  const inlineCodes = [];
  h = h.replace(/`([^`]+)`/g, (match, code) => {
    const placeholder = `__INLINE_CODE_PLACEHOLDER_${inlineCodes.length}__`;
    inlineCodes.push(`<code style="background:var(--accent-dim);padding:2px 6px;border-radius:4px;font-size:13px;color:var(--accent)">${code}</code>`);
    return placeholder;
  });

  // 3. Parse Tables
  h = h.replace(/(?:^|\n)(\|.*\|)(?:\n\|.*\|)+/g, (tableBlock) => {
    const lines = tableBlock.trim().split('\n');
    let hasHeader = false;
    let headerHtml = '';
    let bodyHtml = '';
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      if (line.match(/^\|[ \t]*:?-+:?[ \t]*\|[ \t]*:?-+:?[ \t]*\|/) || line.match(/^\|[ \t]*:?-+:?[ \t]*\|/)) {
        hasHeader = true;
        continue;
      }
      const cells = line.split('|').slice(1, -1).map(c => c.trim());
      const rowHtml = `<tr>${cells.map(c => `<td style="padding:8px 12px;border:1px solid var(--border);">${c}</td>`).join('')}</tr>`;
      
      if (i === 0 && !hasHeader) {
        headerHtml = `<thead><tr>${cells.map(c => `<th style="padding:8px 12px;border:1px solid var(--border);background:var(--bg3);font-weight:600;">${c}</th>`).join('')}</tr></thead>`;
      } else if (i === 0 || (i === 1 && hasHeader)) {
        headerHtml = `<thead><tr>${cells.map(c => `<th style="padding:8px 12px;border:1px solid var(--border);background:var(--bg3);font-weight:600;">${c}</th>`).join('')}</tr></thead>`;
      } else {
        bodyHtml += rowHtml;
      }
    }
    return `\n<table style="width:100%;border-collapse:collapse;margin:12px 0;font-size:13px;background:var(--bg2);border:1px solid var(--border);">${headerHtml}<tbody>${bodyHtml}</tbody></table>\n`;
  });

  // 4. Parse Blockquotes
  h = h.replace(/(?:^|\n)&gt;[ \t]+(.*)/g, (match, quoteText) => {
    return `\n<blockquote style="border-left:3px solid var(--accent);padding-left:12px;color:var(--text2);margin:8px 0;font-style:italic;">${quoteText}</blockquote>\n`;
  });

  // 5. Parse Lists
  h = h.replace(/(?:^|\n)[-*+][ \t]+(.*)/g, '\n<li>$1</li>');
  h = h.replace(/(?:<li>.*<\/li>\s*)+/g, (match) => `<ul style="margin:8px 0 8px 20px;padding-left:8px;list-style-type:disc;">${match.trim()}</ul>`);
  h = h.replace(/(?:^|\n)\d+\.[ \t]+(.*)/g, '\n<li class="ord-li">$1</li>');
  h = h.replace(/(?:<li class="ord-li">.*<\/li>\s*)+/g, (match) => {
    const cleaned = match.replace(/class="ord-li"/g, '');
    return `<ol style="margin:8px 0 8px 20px;padding-left:8px;list-style-type:decimal;">${cleaned.trim()}</ol>`;
  });

  // 6. Horizontal Rules
  h = h.replace(/(?:^|\n)---[ \t]*(\n|$)/g, '\n<hr style="border:0;border-top:1px solid var(--border);margin:16px 0;">\n');

  // 7. Bold and Italic
  h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // 8. Convert newlines to breaks
  h = h.replace(/\n/g, '<br>');
  h = h.replace(/<br><(ul|ol|table|blockquote|pre|hr|li|thead|tbody|tr)/g, '<$1');
  h = h.replace(/<\/(ul|ol|table|blockquote|pre|hr|li|thead|tbody|tr)><br>/g, '</$1>');

  // 9. Restore placeholders
  inlineCodes.forEach((codeHtml, idx) => {
    h = h.replace(`__INLINE_CODE_PLACEHOLDER_${idx}__`, codeHtml);
  });
  codeBlocks.forEach((codeHtml, idx) => {
    h = h.replace(`__CODE_BLOCK_PLACEHOLDER_${idx}__`, codeHtml);
  });

  return h;
}

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}
