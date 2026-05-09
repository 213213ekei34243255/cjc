(function(){

  document.addEventListener('DOMContentLoaded', () => {
    try {
      console.log('[VERONICA WIDGET] init');

      const thisScript = document.currentScript ||
        document.querySelector('script[src*="widget.js"]') ||
        Array.from(document.getElementsByTagName('script')).reverse().find(s=>s.src && s.src.includes('widget.js'));
      const BASE_API = (thisScript && (thisScript.dataset && thisScript.dataset.api)) ||
                       (thisScript && thisScript.getAttribute('data-api')) ||
                       'https://www.byncai.net';
      console.log('[VERONICA WIDGET] BASE_API =', BASE_API);

      // --- persistent session id (works with Redis backend) ---
      function getOrCreateSessionId() {
        try {
          const key = 'veronica_session_id_v1';
          let id = localStorage.getItem(key);

          // if you want a *fresh* chat on every page refresh instead of persisting:
          //   - replace localStorage with sessionStorage
          //   - or just always generate a new id here
          if (!id) {
            id = (window.crypto && crypto.randomUUID)
              ? crypto.randomUUID()
              : 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2,10);
            localStorage.setItem(key, id);
          }
          return id;
        } catch (e) {
          return 'sess-fallback-' + Date.now();
        }
      }
      const SESSION_ID = getOrCreateSessionId();
      // --- end session id ---

      // --- create host element and attach shadow root to isolate styles ---
      const host = document.createElement('div');
      host.id = 'vai-widget-host';
      // keep host out of visual flow (container inside shadow will be positioned)
      host.style.all = 'initial'; // reset host's own styling influence
      // attach shadow root
      const shadow = host.attachShadow({ mode: 'open' });
      // append host to body
      document.body.appendChild(host);

      // CSS moved into shadow (note: replaced :root with :host to scope CSS variables)
      const css = `
        :host{
          --primary: #00d4ff;
          --primary-glow: rgba(0, 212, 255, 0.35);
          --accent: #7b5ea7;
          --accent2: #ff4ecd;
          --bg-deep: #050a12;
          --bg-panel: rgba(8, 15, 28, 0.94);
          --bg-glass: rgba(255,255,255,0.04);
          --border-subtle: rgba(0,212,255,0.18);
          --border-strong: rgba(0,212,255,0.5);
          --text-main: #e8f4ff;
          --text-muted: #6a8aaa;
          --text-dim: #3a5570;
          --neon: #00d4ff;
          --primaryGradient: linear-gradient(160deg, rgba(12,22,42,0.97) 0%, rgba(6,12,26,0.99) 100%);
          --secondaryGradient: linear-gradient(180deg, rgba(4,10,20,0.9), rgba(2,6,16,0.95));
          --primaryBoxShadow: 0 0 28px rgba(0,212,255,0.4), 0 8px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
          --secondaryBoxShadow: 0 0 0 1px rgba(0,212,255,0.08), 0 24px 64px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.06);
          --header-bg: url("${BASE_API}/static/charlie.jpeg");
        }
        
        /* ================= BUTTON ================= */
        .vai-chatbox{
          position: fixed;
          bottom: 24px;
          right: 24px;
          z-index: 2147483647 !important;
          font-family: 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
        }
        
        .vai-chatbox .chatbox__button{
          text-align: right;
        }
        
        .vai-chatbox .chatbox__button button{
          padding: 0;
          border: 1px solid rgba(0,212,255,0.4);
          outline: none;
          cursor: pointer;
          border-radius: 50%;
          width: 58px;
          height: 58px;
          background: linear-gradient(135deg, #0a2540, #12103a);
          color: var(--primary);
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: var(--primaryBoxShadow);
          transition: transform .25s ease, box-shadow .25s ease;
          animation: vai-float 3s ease-in-out infinite;
          position: relative;
        }
        
        .vai-chatbox .chatbox__button button::before{
          content: '';
          position: absolute;
          inset: -5px;
          border-radius: 50%;
          border: 1px solid rgba(0,212,255,0.15);
          animation: vai-ring-pulse 2s ease-in-out infinite;
          pointer-events: none;
        }
        
        .vai-chatbox .chatbox__button button:hover{
          transform: scale(1.07);
          box-shadow: 0 0 40px rgba(0,212,255,0.6), 0 8px 28px rgba(0,0,0,0.6);
        }
        
        @keyframes vai-float{
          0%, 100% { transform: translateY(0); }
          50%       { transform: translateY(-4px); }
        }
        @keyframes vai-ring-pulse{
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50%       { opacity: 0; transform: scale(1.25); }
        }
        
        /* ================= CONTAINER ================= */
        .vai-chatbox .chatbox__support{
          display: flex;
          flex-direction: column;
          position: fixed;
          bottom: 94px;
          right: 24px;
          width: 360px;
          height: 490px;
          border-radius: 20px;
          overflow: hidden;
          background: linear-gradient(160deg, rgba(12,22,42,0.97) 0%, rgba(6,12,26,0.99) 100%);
          border: 1px solid var(--border-subtle);
          transform: translateY(14px) scale(0.98);
          opacity: 0;
          pointer-events: none;
          transition: transform .3s cubic-bezier(0.34,1.4,0.64,1), opacity .25s ease;
          box-shadow: var(--secondaryBoxShadow);
        }
        
        /* Top edge scan-line accent */
        .vai-chatbox .chatbox__support::before{
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent 0%, var(--primary) 40%, var(--accent2) 70%, transparent 100%);
          opacity: 0.75;
          z-index: 10;
        }
        
        /* Subtle grid overlay */
        .vai-chatbox .chatbox__support::after{
          content: '';
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
          background-size: 36px 36px;
          pointer-events: none;
          z-index: 0;
        }
        
        .vai-chatbox .chatbox--active{
          transform: translateY(0) scale(1);
          opacity: 1;
          pointer-events: auto;
        }
        
        /* ================= HEADER ================= */
        .vai-chatbox .chatbox__header{
          position: relative;
          z-index: 2;
          background: linear-gradient(180deg, rgba(0,40,80,0.55) 0%, rgba(0,16,36,0.3) 100%);
          border-bottom: 1px solid var(--border-subtle);
          color: var(--text-main);
          padding: 14px 16px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .vai-chatbox .chatbox__image--header img{
          width: 40px;
          height: 40px;
          border-radius: 50%;
          object-fit: cover;
          box-shadow: 0 0 14px rgba(0,212,255,0.45);
          border: 1px solid var(--border-subtle);
        }
        
        .vai-chatbox .chatbox__heading--header{
          font-size: 14px;
          font-weight: 600;
          margin: 0;
          color: var(--text-main);
          letter-spacing: 0.3px;
        }
        
        .vai-chatbox .chatbox__description--header{
          font-size: 11px;
          color: #1aff90;
          margin: 2px 0 0;
          letter-spacing: 0.5px;
          display: flex;
          align-items: center;
          gap: 5px;
        }
        
        /* Online pulse dot — add a span.vai-online inside .chatbox__description--header */
        .vai-chatbox .vai-online{
          display: inline-block;
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #1aff90;
          box-shadow: 0 0 7px #1aff90;
          animation: vai-pulse 2s infinite;
        }
        
        @keyframes vai-pulse{
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.35; }
        }
        
        /* ================= MESSAGES ================= */
        .vai-chatbox .chatbox__messages{
          position: relative;
          z-index: 1;
          padding: 14px;
          flex: 1;
          overflow: auto;
          display: flex;
          flex-direction: column-reverse;
          gap: 10px;
          background:
            radial-gradient(ellipse at top left, rgba(0,212,255,0.06) 0%, transparent 55%),
            radial-gradient(ellipse at bottom right, rgba(123,94,167,0.07) 0%, transparent 50%),
            rgba(4, 8, 20, 0.85);
          scrollbar-width: thin;
          scrollbar-color: rgba(0,212,255,0.2) transparent;
        }
        
        /* Bot bubble */
        .vai-chatbox .messages__item{
          max-width: 72%;
          padding: 10px 14px;
          border-radius: 16px 16px 16px 4px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(0,212,255,0.12);
          backdrop-filter: blur(8px);
          color: var(--text-main);
          align-self: flex-start;
          word-wrap: break-word;
          white-space: pre-wrap;
          font-size: 13px;
          line-height: 1.55;
          box-shadow: none;
        }
        
        /* User bubble */
        .vai-chatbox .messages__item--operator{
          border-radius: 16px 16px 4px 16px;
          background: linear-gradient(135deg, rgba(0,120,180,0.3), rgba(123,94,167,0.25));
          border-color: rgba(0,212,255,0.25);
          color: var(--text-main);
          align-self: flex-end;
          box-shadow: 0 0 14px rgba(0,212,255,0.1);
        }
        
        /* ================= FOOTER ================= */
        .vai-chatbox .chatbox__footer{
          position: relative;
          z-index: 2;
          padding: 12px 14px;
          background: rgba(4,10,20,0.88);
          display: flex;
          gap: 8px;
          align-items: center;
          border-top: 1px solid var(--border-subtle);
        }
        
        .vai-chatbox .chatbox__footer input{
          flex: 1;
          padding: 10px 14px;
          border-radius: 12px;
          border: 1px solid var(--border-subtle);
          background: rgba(255,255,255,0.04);
          color: var(--text-main);
          font-size: 13px;
          outline: none;
          transition: border-color .2s, box-shadow .2s;
        }
        
        .vai-chatbox .chatbox__footer input::placeholder{
          color: var(--text-dim);
        }
        
        .vai-chatbox .chatbox__footer input:focus{
          border-color: var(--border-strong);
          box-shadow: 0 0 0 3px rgba(0,212,255,0.08);
        }
        
        .vai-chatbox .chatbox__footer button{
          width: 38px;
          height: 38px;
          flex-shrink: 0;
          border-radius: 12px;
          border: 1px solid rgba(0,212,255,0.4);
          background: linear-gradient(135deg, rgba(0,180,255,0.18), rgba(123,94,167,0.22));
          color: var(--primary);
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          box-shadow: 0 0 12px rgba(0,212,255,0.15), inset 0 1px 0 rgba(255,255,255,0.08);
          transition: background .2s, box-shadow .2s, transform .15s;
        }
        
        .vai-chatbox .chatbox__footer button:hover{
          background: linear-gradient(135deg, rgba(0,212,255,0.32), rgba(123,94,167,0.38));
          box-shadow: 0 0 22px rgba(0,212,255,0.35);
          transform: scale(1.06);
        }
        
        /* ================= STATUS & LINKS ================= */
        .vai-chatbox .status{
          padding: 8px 12px;
          color: var(--text-dim);
          font-size: 11px;
          text-align: center;
          letter-spacing: 0.3px;
        }
        
        .vai-chatbox a.vai-link{
          color: var(--primary);
          text-decoration: underline;
          word-break: break-all;
        }
        
        /* ================= COPYRIGHT ================= */
        .vai-chatbox .chatbox__copyright{
          position: relative;
          z-index: 2;
          background: rgba(2,6,16,0.9);
          border-top: 1px solid rgba(0,212,255,0.08);
          color: var(--text-dim);
          font-size: 10px;
          text-align: center;
          padding: 5px 0;
          letter-spacing: 0.5px;
        }
        
        .vai-chatbox .chatbox__copyright a{
          color: var(--text-dim);
          text-decoration: none;
          transition: color .2s;
        }
        
        .vai-chatbox .chatbox__copyright a:hover{
          color: var(--primary);
        }
        
        .vai-chatbox .chatbox__button i{
          font-size: 26px;
          line-height: 1;
        }
        `;

      // create style element and append to shadow root (isolated)
      const styleEl = document.createElement('style');
      styleEl.setAttribute('type','text/css');
      styleEl.appendChild(document.createTextNode(css));
      shadow.appendChild(styleEl);

      // Create widget container inside the shadow root
      const container = document.createElement('div');
      container.className = 'vai-chatbox';
      container.innerHTML = `
        <div class="chatbox__support" id="vai_support" aria-hidden="true">
          <div class="chatbox__header">
            <div class="chatbox__image--header"><img src="${BASE_API}/static/logo.png" alt="VAI" /></div>
            <div class="chatbox__content--header">
              <h4 class="chatbox__heading--header">Noah</h4>
              <p class="chatbox__description--header">Institutional Language Model</p>
            </div>
          </div>
          <div class="chatbox__messages" id="vai_messages" aria-live="polite"></div>
          <div class="chatbox__copyright">
            © 2026 <a href="https://www.cogniaistudios.com" target="_blank" rel="noopener noreferrer">CogniAI Studios</a>. All rights reserved.
          </div>
          <div class="chatbox__footer">
            <input id="vai_input" type="text" placeholder="Write a message..." aria-label="Message" />
            <button id="vai_send" type="button">Send</button>
          </div>
          <div class="status" id="vai_status" style="display:none"></div>
        </div>
        <div class="chatbox__button">
          <button id="vai_toggle" title="Chat with Veronica">
            <i class='bx bxs-message'></i>
          </button>
        </div>
      `;
      // append container to shadow root (so DOM + styles are encapsulated)
      shadow.appendChild(container);

      // --- START: external floating button (kept exactly as you had it) ---
      const originalToggleBtn = shadow.getElementById('vai_toggle');
      if (originalToggleBtn) {
        originalToggleBtn.style.display = 'none';
      }

      const externalToggle = document.createElement('button');
      externalToggle.id = 'vai_external_toggle';
      externalToggle.innerHTML = '💬';
      
      Object.assign(externalToggle.style, {
        position: 'fixed',
        right: '20px',
        bottom: '20px',
        width: '60px',
        height: '60px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: '0 6px 14px rgba(0,0,0,0.25)',
        border: '0',
      
        /* 🔥 IMAGE BACKGROUND */
        background: '#17f38c',
      
        color: '#fff',
        cursor: 'pointer',
        zIndex: '2147483001',
        fontSize: '22px',
        outline: 'none'
      });
      
      document.body.appendChild(externalToggle);

      // --- END replacement ---

      // Query elements inside shadow root (isolation preserved)
      const support = shadow.getElementById('vai_support');
      const toggle = shadow.getElementById('vai_toggle'); // original toggle inside shadow (hidden)
      const messagesEl = shadow.getElementById('vai_messages');
      const inputEl = shadow.getElementById('vai_input');
      const sendBtn = shadow.getElementById('vai_send');
      const statusEl = shadow.getElementById('vai_status');

      function setStatus(msg, show=true) {
        if (!msg) {
          statusEl.style.display='none';
          statusEl.textContent='';
          return;
        }
        statusEl.style.display = show ? 'block' : 'none';
        statusEl.textContent = msg;
      }

      // toggle behavior (shadow button + external button)
      if (toggle) {
        try {
          toggle.addEventListener('click', () => {
            support.classList.toggle('chatbox--active');
            if (support.classList.contains('chatbox--active')) inputEl.focus();
          });
        } catch(e) { console.warn('Failed to attach listener to original toggle', e); }
      }

      externalToggle.addEventListener('click', () => {
        support.classList.toggle('chatbox--active');
        if (support.classList.contains('chatbox--active')) inputEl.focus();
      });

      function appendMessage(text, who='veronica') {
        const div = document.createElement('div');
        div.className = 'messages__item ' + (who === 'you' ? 'messages__item--operator' : 'messages__item--visitor');
        div.textContent = text;

        messagesEl.insertBefore(div, messagesEl.firstChild);

        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
      }

      function appendHtmlMessage(htmlContent, who='veronica') {
        const div = document.createElement('div');
        div.className = 'messages__item ' + (who === 'you' ? 'messages__item--operator' : 'messages__item--visitor');
        div.innerHTML = htmlContent;
        messagesEl.insertBefore(div, messagesEl.firstChild);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
      }

      function openUrlInNewTab(url) {
        try {
          const newWin = window.open(url, '_blank', 'noopener,noreferrer');
          if (!newWin) {
            appendHtmlMessage(`<a class="vai-link" href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`, 'veronica');
          }
        } catch (e) {
          console.error('[VERONICA] open url failed', e);
          appendHtmlMessage(`<a class="vai-link" href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`, 'veronica');
        }
      }

      async function sendMessage() {
        const val = inputEl.value.trim();
        if (!val) return;
        const userDiv = appendMessage(val, 'you');
        inputEl.value = '';
        setStatus('Sending...');
        try {

          const typingDiv = appendMessage('Just a moment...', 'veronica');

          const doFetch = async () => {
            const payload = {
              message: val,
              session_id: SESSION_ID,   // <--- this is what your Redis backend uses
              url: location.href,
              user_agent: navigator.userAgent || '',
              language: navigator.language || '',
              timestamp: (new Date()).toISOString()
            };
            console.log('[VERONICA] POST', BASE_API + '/predict', 'payload:', payload);
            const res = await fetch(BASE_API + '/predict', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(payload),
              keepalive: true
            });
            return res;
          };

          let res = await doFetch();

          if (typingDiv && typingDiv.parentNode) messagesEl.removeChild(typingDiv);

          if (!res.ok) {
            console.warn('[VERONICA] first fetch not ok, status=', res.status);
            setStatus('Veronica is waking up — retrying in a few seconds...', true);
            await new Promise(r=>setTimeout(r, 3500));
            res = await doFetch();
            if (!res.ok) {
              console.error('[VERONICA] retry failed, status=', res.status);
              setStatus(`Server error (${res.status}). Try again later.`);
              return;
            }
          }

          let data;
          try {
            data = await res.json();
          } catch (e) {
            console.error('[VERONICA] failed to parse JSON', e);
            setStatus('Invalid response from server.');
            return;
          }

          if (data.url) {
            appendMessage(data.answer || (`Opening: ${data.url}`), 'veronica');
            openUrlInNewTab(data.url);
            setStatus('');
            if (data.reply_id) console.log('[VERONICA] reply_id:', data.reply_id);
            return;
          }

          const botDiv = appendMessage(data.answer || 'Sorry, no reply', 'veronica');
          if (data.reply_id) botDiv.dataset.replyId = data.reply_id;
          if (data.reply_id) console.log('[VERONICA] reply_id:', data.reply_id);

          setStatus('');
        } catch (err) {
          console.error('Widget fetch error', err);

          if (err instanceof TypeError) {
            setStatus('Network error or CORS blocked. Check console and server CORS settings.');
          } else {
            setStatus('Error connecting to Veronica. Try again later.');
          }
        }
      }

      sendBtn.addEventListener('click', sendMessage);
      inputEl.addEventListener('keyup', (e) => { if (e.key === 'Enter') sendMessage(); });

      // expose API that operates on elements inside shadow root
      window.VERONICA_WIDGET = {
        baseUrl: BASE_API,
        sessionId: SESSION_ID,
        open: () => support.classList.add('chatbox--active'),
        close: () => support.classList.remove('chatbox--active'),
        send: (msg) => { inputEl.value = msg; sendMessage(); }
      };

      console.log('[VERONICA WIDGET] ready (shadow DOM enabled)');
    } catch (e) {
      console.error('[VERONICA WIDGET] init failed', e);
    }
  });
})();

















