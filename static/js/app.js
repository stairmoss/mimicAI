const API = {
  chat: '/api/chat',
  voices: '/api/voices',
  voiceRecord: '/api/voices/record',
  voiceDelete: (id) => `/api/voices/${id}/delete`,
  voicePreview: (id) => `/api/voices/${id}/preview`,
  voiceGender: (id) => `/api/voices/${id}/gender`,
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
  voiceMode: 'mimicai:voice-mode',
  voiceDesign: 'mimicai:voice-design',
};

const state = {
  messages: [],
  voices: [],
  selectedVoice: null,
  isGenerating: false,
  mediaRecorder: null,
  recordedChunks: [],
  recordingTimer: null,
  recordingSeconds: 0,
  autoSpeak: true,
  lightweightTts: false,
  currentAudio: null,
  theme: "dark",
  voiceMode: 'profiles',
  voiceDesign: {
    gender: 'male',
    accent: 'us',
    speed: 1.0,
    pitch: 1.0,
  },
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

document.addEventListener('DOMContentLoaded', () => {
  state.selectedVoice = localStorage.getItem(KEYS.voice) || null;
  const savedAutoSpeak = localStorage.getItem(KEYS.autoSpeak);
  state.autoSpeak = savedAutoSpeak !== null ? savedAutoSpeak === 'true' : true;
  state.lightweightTts = localStorage.getItem(KEYS.lightweightTts) === 'true';
  state.theme = localStorage.getItem(KEYS.theme) || 'dark';
  document.documentElement.setAttribute('data-theme', state.theme);

  state.voiceMode = localStorage.getItem(KEYS.voiceMode) || 'profiles';
  try {
    const saved = localStorage.getItem(KEYS.voiceDesign);
    if (saved) state.voiceDesign = JSON.parse(saved);
  } catch {}

  bindEvents();
  applyVoiceModeUI();
  applyVoiceDesignUI();
  loadVoices();
  loadLanguages();
  checkEngineStatus();
  setInterval(checkEngineStatus, 30000);
});


function primeAudio() {
  const sa = $('#speech-audio');
  if (sa) {
    if (!sa.src) {
      sa.src = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAA==';
    }
    sa.play().then(() => { sa.pause(); }).catch(() => {});
  }
  if (window._audioContext && window._audioContext.state === 'suspended') {
    window._audioContext.resume().catch(() => {});
  }
}

function bindEvents() {
  window.addEventListener('pointerdown', primeAudio, { once: true });
  window.addEventListener('keydown', primeAudio, { once: true });
  $('#send-btn').addEventListener('click', sendMessage);
  $('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $('#chat-input').addEventListener('input', () => {
    const el = $('#chat-input');
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
  });

  $('#toggle-sidebar').addEventListener('click', () => $('.sidebar').classList.toggle('hidden'));
  const nb = $('#new-chat-btn');
  if (nb) nb.addEventListener('click', newChat);

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
      toast(state.lightweightTts ? 'Lightweight mode on' : 'Full mode on', 'info');
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
        toast('Speech playback stopped', 'info');
      }
    });
  }

  const themeBtn = $('#theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
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
      const val = ta.value;
      ta.value = val.substring(0, start) + sym + val.substring(end);
      ta.focus();
      ta.selectionStart = ta.selectionEnd = start + sym.length;
      toast('Inserted ' + sym, 'info');
      ta.dispatchEvent(new Event('input'));
    });
  });

  const tabP = $('#tab-profiles');
  const tabD = $('#tab-design');
  if (tabP && tabD) {
    tabP.addEventListener('click', () => {
      state.voiceMode = 'profiles';
      localStorage.setItem(KEYS.voiceMode, 'profiles');
      applyVoiceModeUI();
    });
    tabD.addEventListener('click', () => {
      state.voiceMode = 'design';
      localStorage.setItem(KEYS.voiceMode, 'design');
      applyVoiceModeUI();
    });
  }

  $$('.design-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.voiceDesign.gender = btn.dataset.gender;
      localStorage.setItem(KEYS.voiceDesign, JSON.stringify(state.voiceDesign));
      applyVoiceDesignUI();
    });
  });

  const accentEl = $('#design-accent');
  if (accentEl) {
    accentEl.addEventListener('change', (e) => {
      state.voiceDesign.accent = e.target.value;
      localStorage.setItem(KEYS.voiceDesign, JSON.stringify(state.voiceDesign));
    });
  }

  const speedEl = $('#design-speed');
  if (speedEl) {
    speedEl.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value).toFixed(1);
      state.voiceDesign.speed = Math.max(0.5, Math.min(2.0, parseFloat(v)));
      $('#val-speed').textContent = v + 'x';
      localStorage.setItem(KEYS.voiceDesign, JSON.stringify(state.voiceDesign));
    });
  }

  const pitchEl = $('#design-pitch');
  if (pitchEl) {
    pitchEl.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value).toFixed(1);
      state.voiceDesign.pitch = parseFloat(v);
      $('#val-pitch').textContent = v;
      localStorage.setItem(KEYS.voiceDesign, JSON.stringify(state.voiceDesign));
    });
  }
}

async function sendMessage() {
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text || state.isGenerating) return;

  primeAudio();

  addMsg('user', text);
  input.value = '';
  input.style.height = 'auto';
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
    let aiText = '';
    let buf = '';
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
      if (state.autoSpeak) {
        if (!state.selectedVoice && state.voices.length > 0) {
          state.selectedVoice = state.voices[0].id;
          persistVoice();
          updateVoiceSelect();
        }
        const btn = msgEl.querySelector('.tts-btn');
        playTts(aiText, btn, true);
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
    <div class="message-avatar">${isAi ? 'AI' : 'U'}</div>
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

  const tb = document.createElement('button');
  tb.className = 'msg-action-btn tts-btn';
  tb.innerHTML = 'Listen';
  tb.onclick = () => playTts(text, tb);
  a.appendChild(tb);

  const cb = document.createElement('button');
  cb.className = 'msg-action-btn';
  cb.innerHTML = 'Copy';
  cb.onclick = () => {
    navigator.clipboard.writeText(text).then(() => {
      cb.innerHTML = 'Copied';
      setTimeout(() => { cb.innerHTML = 'Copy'; }, 2000);
    });
  };
  a.appendChild(cb);
}

function showTyping() {
  const c = $('#messages');
  const el = document.createElement('div');
  el.className = 'typing-indicator';
  el.id = 'typing-indicator';
  el.innerHTML = `
    <div class="message-avatar">AI</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>`;
  c.appendChild(el);
  scroll();
}

function removeTyping() {
  const el = $('#typing-indicator');
  if (el) el.remove();
}

function hideWelcome() {
  const w = $('#welcome-screen');
  if (w) w.style.display = 'none';
}

function newChat() {
  state.messages = [];
  const c = $('#messages');
  c.innerHTML = `
    <div class="welcome" id="welcome-screen">
      <div class="welcome-glow"></div>
      <div class="welcome-logo">
        <svg width="72" height="72" viewBox="0 0 68 68" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="welWave1" x1="12" y1="34" x2="56" y2="34" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#8ab4f8"/>
              <stop offset="100%" stop-color="#80d8ff"/>
            </linearGradient>
            <linearGradient id="welWave2" x1="12" y1="34" x2="56" y2="34" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#aecbfa"/>
              <stop offset="100%" stop-color="#8ab4f8"/>
            </linearGradient>
            <radialGradient id="welGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="rgba(138, 180, 248, 0.35)"/>
              <stop offset="100%" stop-color="rgba(138, 180, 248, 0)"/>
            </radialGradient>
          </defs>
          <rect width="68" height="68" rx="19" fill="#292a2d" stroke="#3c4043" stroke-width="1.6"/>
          <circle cx="34" cy="34" r="28" fill="url(#welGlow)"/>
          <line x1="17.5" y1="26" x2="17.5" y2="42" stroke="#8ab4f8" stroke-width="2" stroke-linecap="round" opacity="0.35"/>
          <line x1="23" y1="19" x2="23" y2="49" stroke="#8ab4f8" stroke-width="2.5" stroke-linecap="round" opacity="0.45"/>
          <line x1="45" y1="19" x2="45" y2="49" stroke="#80d8ff" stroke-width="2.5" stroke-linecap="round" opacity="0.45"/>
          <line x1="50.5" y1="26" x2="50.5" y2="42" stroke="#80d8ff" stroke-width="2" stroke-linecap="round" opacity="0.35"/>
          <path d="M12 34 C17 16, 29 16, 34 34 C39 52, 51 52, 56 34" stroke="url(#welWave1)" stroke-width="3.5" stroke-linecap="round"/>
          <path d="M12 34 C17 52, 29 52, 34 34 C39 16, 51 16, 56 34" stroke="url(#welWave2)" stroke-width="3.5" stroke-linecap="round"/>
          <circle cx="23" cy="17" r="3.2" fill="#8ab4f8"/>
          <circle cx="23" cy="51" r="3.2" fill="#aecbfa"/>
          <circle cx="45" cy="17" r="3.2" fill="#80d8ff"/>
          <circle cx="45" cy="51" r="3.2" fill="#8ab4f8"/>
          <circle cx="12" cy="34" r="2.2" fill="#8ab4f8"/>
          <circle cx="56" cy="34" r="2.2" fill="#80d8ff"/>
          <path d="M34 27 Q34 34 27 34 Q34 34 34 41 Q34 34 41 34 Q34 34 34 27 Z" fill="#ffffff"/>
          <circle cx="34" cy="34" r="1.6" fill="#8ab4f8"/>
        </svg>
      </div>
      <h1>MimicAI</h1>
      <p>Chat with AI and hear responses spoken aloud. Record a voice sample in the sidebar, then start chatting.</p>
      <div class="welcome-pills">
        <div class="welcome-pill" data-prompt="Tell me a joke">Tell me a joke</div>
        <div class="welcome-pill" data-prompt="Explain quantum physics simply">Explain quantum physics</div>
        <div class="welcome-pill" data-prompt="Write a short poem about the ocean">Write a poem</div>
        <div class="welcome-pill" data-prompt="What can you do?">What can you do?</div>
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
  const btn = $('#send-btn');
  btn.disabled = loading;
  if (loading) btn.innerHTML = '<div class="tts-spinner" style="border-top-color:#202124"></div>';
  else btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
}

function scroll() {
  const el = $('#chat-area');
  requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
}

async function playTts(text, btn, isAutoSpeak = false) {
  if (!text) return;
  let isProfiles = state.voiceMode === 'profiles';
  let voiceId = isProfiles ? (state.selectedVoice || null) : null;
  if (isProfiles && !voiceId) {
    if (state.voices.length > 0) {
      voiceId = state.voices[0].id;
      state.selectedVoice = voiceId;
      persistVoice();
      updateVoiceSelect();
    } else {
      isProfiles = false;
    }
  }

  const voiceDesign = isProfiles ? null : state.voiceDesign;
  let lang = 'en';
  if (isProfiles) {
    const prof = state.voices.find(v => v.id === voiceId);
    lang = prof?.language || 'en';
  } else {
    lang = voiceDesign?.accent === 'jp' ? 'ja' : 'en';
  }

  const orig = btn ? btn.innerHTML : 'Listen';
  if (btn) {
    btn.innerHTML = '⏳ Loading…';
    btn.disabled = true;
  }

  let statusPill = btn?.parentElement?.querySelector('.speaking-status-pill');
  if (!statusPill && btn?.parentElement) {
    statusPill = document.createElement('div');
    statusPill.className = 'speaking-status-pill loading';
    statusPill.innerHTML = '<span>⏳ Synthesizing voice…</span>';
    btn.parentElement.appendChild(statusPill);
  }

  try {
    const res = await fetch(API.tts, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text.substring(0, 500),
        voice_id: voiceId,
        language: lang,
        prefer_clone: isProfiles,
        lightweight: state.lightweightTts,
        strict_clone: false,
        voice_design: voiceDesign,
        speed_rate: state.voiceDesign?.speed || 1.0,
      }),
    });

    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.error || 'TTS failed');
    }

    const engine = res.headers.get('X-TTS-Engine');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    let playerBox = btn?.parentElement?.querySelector('.audio-player-container');
    if (!playerBox && btn?.parentElement) {
      playerBox = document.createElement('div');
      playerBox.className = 'audio-player-container';
      playerBox.style.marginTop = '10px';
      playerBox.style.width = '100%';
      btn.parentElement.appendChild(playerBox);
    }

    if (playerBox) {
      playerBox.innerHTML = `<audio controls src="${url}" style="height:32px; width:100%; outline:none; border-radius:16px;"></audio>`;
    }

    const inlineAudio = playerBox?.querySelector('audio');
    const speechAudio = $('#speech-audio');
    const activeAudio = speechAudio || inlineAudio;

    if (statusPill) {
      statusPill.className = 'speaking-status-pill';
      statusPill.innerHTML = `
        <span class="speaking-wave"><span></span><span></span><span></span><span></span></span>
        <span>🔊 Speaking aloud…</span>
      `;
    }

    if (speechAudio) {
      speechAudio.src = url;
    }

    setCurrentAudio(activeAudio);

    const onFinish = () => {
      if (statusPill) statusPill.remove();
      if (btn) {
        btn.innerHTML = '▶ Replay Speech';
        btn.disabled = false;
      }
    };

    activeAudio.onended = onFinish;
    activeAudio.onpause = () => {
      if (statusPill && activeAudio.ended) statusPill.remove();
    };

    if (btn) {
      btn.innerHTML = '🔊 Speaking…';
      btn.disabled = false;
    }

    try {
      await activeAudio.play();
    } catch (e) {
      console.warn('Autoplay prevented:', e);
      if (inlineAudio && inlineAudio !== activeAudio) {
        try { await inlineAudio.play(); } catch (_) {}
      }
      if (statusPill) {
        statusPill.className = 'speaking-status-pill';
        statusPill.innerHTML = '<span>▶ Click Listen to hear speech</span>';
      }
    }

    if (engine) toast(`Generated via ${engine}`, 'info');
  } catch (err) {
    if (statusPill) statusPill.remove();
    if (btn) {
      btn.innerHTML = orig;
      btn.disabled = false;
    }
    toast(err.message, 'error');
  }
}

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
    const isFem = (v.gender || '').toLowerCase() === 'female';
    const li = document.createElement('li');
    li.className = `voice-item ${state.selectedVoice === v.id ? 'active' : ''}`;
    li.innerHTML = `
      <div class="voice-avatar">${v.name.charAt(0).toUpperCase()}</div>
      <div class="voice-info">
        <div class="voice-name">${esc(v.name)}</div>
        <div class="voice-lang">
          ${v.language || 'en'} · 
          <span class="voice-gender-pill" title="Click to toggle Male/Female" style="cursor:pointer;padding:1px 6px;border-radius:10px;background:${isFem ? 'rgba(242,139,130,0.2)' : 'rgba(138,180,248,0.2)'};color:${isFem ? '#f28b82' : '#8ab4f8'};border:1px solid ${isFem ? '#f28b82' : '#8ab4f8'};font-size:10px;font-weight:600;">
            ${isFem ? '♀ Female' : '♂ Male'}
          </span>
        </div>
      </div>
      <div class="voice-actions">
        <button class="voice-act-btn preview" title="Preview">▶</button>
        <button class="voice-act-btn delete" title="Delete">✕</button>
      </div>`;
    li.addEventListener('click', (e) => {
      if (e.target.closest('.voice-act-btn') || e.target.closest('.voice-gender-pill')) return;
      state.selectedVoice = v.id;
      $('#voice-select').value = v.id;
      persistVoice();
      renderVoiceList();
      toast(`Voice: ${v.name} (${v.gender || 'male'})`, 'success');
    });
    const gPill = li.querySelector('.voice-gender-pill');
    if (gPill) {
      gPill.addEventListener('click', async (e) => {
        e.stopPropagation();
        const nextGender = isFem ? 'male' : 'female';
        try {
          await fetch(API.voiceGender(v.id), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gender: nextGender }),
          });
          v.gender = nextGender;
          toast(`${v.name} set to ${nextGender}`, 'success');
          renderVoiceList();
        } catch (err) {
          toast('Error changing gender: ' + err.message, 'error');
        }
      });
    }
    li.querySelector('.preview').addEventListener('click', (e) => { e.stopPropagation(); previewVoice(v, e.currentTarget); });
    li.querySelector('.delete').addEventListener('click', (e) => { e.stopPropagation(); deleteVoice(v.id); });
    list.appendChild(li);
  });
}

function updateVoiceSelect() {
  const sel = $('#voice-select');
  while (sel.options.length > 1) sel.remove(1);
  state.voices.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = `🎤 ${v.name}`;
    sel.appendChild(opt);
  });
  if (state.selectedVoice) {
    const ok = state.voices.some(v => v.id === state.selectedVoice);
    if (ok) sel.value = state.selectedVoice;
    else { state.selectedVoice = null; sel.value = ''; persistVoice(); }
  } else if (state.voices.length > 0) {
    state.selectedVoice = state.voices[0].id;
    sel.value = state.selectedVoice;
    persistVoice();
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
    audio.onended = () => {
      if (stopBtn) stopBtn.style.display = 'none';
      setCurrentAudio(null);
      toast('Speech playback stopped', 'info');
    };
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
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '…';
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
    a.onended = () => {
      setCurrentAudio(null);
      toast('Speech playback stopped', 'info');
      URL.revokeObjectURL(url);
    };
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = orig;
  }
}

async function deleteVoice(id) {
  if (!confirm('Delete this voice profile?')) return;
  try {
    const r = await fetch(API.voiceDelete(id), { method: 'DELETE' });
    if (r.ok) {
      if (state.selectedVoice === id) {
        state.selectedVoice = null;
        $('#voice-select').value = '';
        persistVoice();
      }
      toast('Voice deleted', 'success');
      loadVoices();
    }
  } catch {
    toast('Failed to delete', 'error');
  }
}

function openModal() {
  $('#record-modal').classList.add('active');
  resetRecording();
}

function closeModal() {
  $('#record-modal').classList.remove('active');
  stopRecording();
  resetRecording();
}

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
  fd.append('language', lang || 'en');
  fd.append('gender', $('#voice-gender')?.value || 'male');

  const btn = $('#save-voice-btn');
  btn.disabled = true;
  btn.textContent = 'Saving…';

  $('#training-modal').classList.add('active');
  const barEl = $('#training-progress-bar');
  const statusEl = $('#training-status-text');

  let pct = 10;
  barEl.style.width = '10%';
  statusEl.textContent = 'Uploading voice sample...';

  const timer = setInterval(() => {
    if (pct < 90) {
      pct += Math.floor(Math.random() * 8) + 2;
      if (pct > 90) pct = 90;
      barEl.style.width = `${pct}%`;

      if (pct < 30) statusEl.textContent = 'Uploading voice sample...';
      else if (pct < 60) statusEl.textContent = 'Transcribing reference speech patterns (Whisper)...';
      else statusEl.textContent = 'Generating speaker profile and optimizing F5-TTS...';
    }
  }, 1000);

  try {
    const r = await fetch(API.voiceRecord, { method: 'POST', body: fd });
    const d = await r.json().catch(() => ({}));
    clearInterval(timer);

    if (r.ok && d.success) {
      barEl.style.width = '100%';
      statusEl.textContent = 'Training complete!';
      await new Promise(res => setTimeout(res, 800));
      $('#training-modal').classList.remove('active');

      toast(`Voice "${name}" saved!`, 'success');
      closeModal();
      await loadVoices();
      if (d.voice?.id) {
        state.selectedVoice = d.voice.id;
        $('#voice-select').value = d.voice.id;
        persistVoice();
        renderVoiceList();
      }
    } else {
      $('#training-modal').classList.remove('active');
      toast(d.error || 'Save failed', 'error');
    }
  } catch (e) {
    clearInterval(timer);
    $('#training-modal').classList.remove('active');
    toast('Connection error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> Save Voice Profile';
  }
}

async function loadLanguages() {
  try {
    const r = await fetch(API.languages);
    const d = await r.json();
    const langs = d.languages || [];
    const sel = $('#voice-language');
    if (!sel) return;
    const existing = new Set(Array.from(sel.options).map(o => o.value));
    langs.forEach(l => {
      if (!existing.has(l.id)) {
        const o = document.createElement('option');
        o.value = l.id;
        o.textContent = l.name;
        sel.appendChild(o);
      }
    });
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

function toast(msg, type = 'success') {
  const c = $('#toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  const icon = type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ';
  t.innerHTML = `${icon} ${esc(msg)}`;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateX(20px)';
    setTimeout(() => t.remove(), 300);
  }, 3500);
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function md(text) {
  if (!text) return '';
  let h = esc(text);

  const codeBlocks = [];
  h = h.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
    const id = `__CODE_${codeBlocks.length}__`;
    codeBlocks.push(`<pre style="background:rgba(255,255,255,0.04);padding:12px;border-radius:8px;overflow-x:auto;font-size:13px;margin:8px 0;border:1px solid var(--border)"><code>${code}</code></pre>`);
    return id;
  });

  const inlineCodes = [];
  h = h.replace(/`([^`]+)`/g, (m, code) => {
    const id = `__INLINE_${inlineCodes.length}__`;
    inlineCodes.push(`<code style="background:var(--accent-dim);padding:2px 6px;border-radius:4px;font-size:13px;color:var(--accent)">${code}</code>`);
    return id;
  });

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

  h = h.replace(/(?:^|\n)&gt;[ \t]+(.*)/g, (m, txt) => {
    return `\n<blockquote style="border-left:3px solid var(--accent);padding-left:12px;color:var(--text2);margin:8px 0;font-style:italic;">${txt}</blockquote>\n`;
  });

  h = h.replace(/(?:^|\n)[-*+][ \t]+(.*)/g, '\n<li>$1</li>');
  h = h.replace(/(?:<li>.*<\/li>\s*)+/g, (m) => `<ul style="margin:8px 0 8px 20px;padding-left:8px;list-style-type:disc;">${m.trim()}</ul>`);
  h = h.replace(/(?:^|\n)\d+\.[ \t]+(.*)/g, '\n<li class="ord-li">$1</li>');
  h = h.replace(/(?:<li class="ord-li">.*<\/li>\s*)+/g, (m) => {
    const cleaned = m.replace(/class="ord-li"/g, '');
    return `<ol style="margin:8px 0 8px 20px;padding-left:8px;list-style-type:decimal;">${cleaned.trim()}</ol>`;
  });

  h = h.replace(/(?:^|\n)---[ \t]*(\n|$)/g, '\n<hr style="border:0;border-top:1px solid var(--border);margin:16px 0;">\n');
  h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
  h = h.replace(/\n/g, '<br>');
  h = h.replace(/<br><(ul|ol|table|blockquote|pre|hr|li|thead|tbody|tr)/g, '<$1');
  h = h.replace(/<\/(ul|ol|table|blockquote|pre|hr|li|thead|tbody|tr)><br>/g, '</$1>');

  inlineCodes.forEach((codeHtml, idx) => {
    h = h.replace(`__INLINE_${idx}__`, codeHtml);
  });
  codeBlocks.forEach((codeHtml, idx) => {
    h = h.replace(`__CODE_${idx}__`, codeHtml);
  });

  return h;
}

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function applyVoiceModeUI() {
  const isProfiles = state.voiceMode === 'profiles';
  const tabP = $('#tab-profiles');
  const tabD = $('#tab-design');
  const panelP = $('#panel-profiles');
  const panelD = $('#panel-design');

  if (isProfiles) {
    if (tabP) tabP.classList.add('active');
    if (tabD) tabD.classList.remove('active');
    if (panelP) panelP.classList.remove('hidden');
    if (panelD) panelD.classList.add('hidden');
  } else {
    if (tabD) tabD.classList.add('active');
    if (tabP) tabP.classList.remove('active');
    if (panelD) panelD.classList.remove('hidden');
    if (panelP) panelP.classList.add('hidden');
  }
}

function applyVoiceDesignUI() {
  const d = state.voiceDesign;
  $$('.design-toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.gender === d.gender);
  });
  const accentEl = $('#design-accent');
  if (accentEl) accentEl.value = d.accent;

  const speedEl = $('#design-speed');
  if (speedEl) speedEl.value = d.speed;
  const valSpeed = $('#val-speed');
  if (valSpeed) valSpeed.textContent = d.speed + 'x';

  const pitchEl = $('#design-pitch');
  if (pitchEl) pitchEl.value = d.pitch;
  const valPitch = $('#val-pitch');
  if (valPitch) valPitch.textContent = d.pitch;
}
