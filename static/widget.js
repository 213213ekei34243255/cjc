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

      // --- persistent session id ---
      function getOrCreateSessionId() {
        try {
          const key = 'veronica_session_id_v1';
          let id = localStorage.getItem(key);
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

      // --- create host element and attach shadow root ---
      const host = document.createElement('div');
      host.id = 'vai-widget-host';
      host.style.all = 'initial';
      const shadow = host.attachShadow({ mode: 'open' });
      document.body.appendChild(host);

      // =====================================================
      // 3D NOAH UI — CSS (merged from your CSS + 3D UI)
      // =====================================================
      const css = `
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700&display=swap');

        :host {
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

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        /* ================= FLOATING TOGGLE BUTTON ================= */
        .vai-chatbox {
          position: fixed;
          bottom: 24px;
          right: 24px;
          z-index: 2147483647 !important;
          font-family: 'Exo 2', 'Segoe UI', Arial, sans-serif;
        }

        .vai-chatbox .chatbox__button { text-align: right; }

        .vai-chatbox .chatbox__button button {
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

        .vai-chatbox .chatbox__button button::before {
          content: '';
          position: absolute;
          inset: -5px;
          border-radius: 50%;
          border: 1px solid rgba(0,212,255,0.15);
          animation: vai-ring-pulse 2s ease-in-out infinite;
          pointer-events: none;
        }

        .vai-chatbox .chatbox__button button:hover {
          transform: scale(1.07);
          box-shadow: 0 0 40px rgba(0,212,255,0.6), 0 8px 28px rgba(0,0,0,0.6);
        }

        @keyframes vai-float {
          0%, 100% { transform: translateY(0); }
          50%       { transform: translateY(-4px); }
        }
        @keyframes vai-ring-pulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50%       { opacity: 0; transform: scale(1.25); }
        }

        /* ================= 3D OUTER SHELL ================= */
        .chatbox__support {
          display: flex;
          flex-direction: column;
          position: fixed;
          bottom: 94px;
          right: 24px;
          width: 380px;
          height: 540px;
          opacity: 0;
          pointer-events: none;
          transform: translateY(14px) scale(0.97) rotateX(4deg) rotateY(-2deg);
          transform-style: preserve-3d;
          transition: transform .35s cubic-bezier(0.34,1.4,0.64,1), opacity .25s ease;
          filter: drop-shadow(0 40px 80px rgba(0,180,255,0.18)) drop-shadow(0 0 60px rgba(0,100,200,0.1));
          perspective: 1000px;
        }

        /* 3D top face */
        .chatbox__support::before {
          content: '';
          position: absolute;
          top: -10px; left: 10px; right: 10px; height: 10px;
          background: linear-gradient(180deg, rgba(0,180,255,0.35), rgba(0,100,160,0.15));
          transform: rotateX(90deg);
          transform-origin: bottom center;
          border-radius: 4px 4px 0 0;
          border-top: 1px solid rgba(0,220,255,0.6);
          z-index: 0;
        }

        /* 3D bottom face */
        .chatbox__support::after {
          content: '';
          position: absolute;
          bottom: -10px; left: 10px; right: 10px; height: 10px;
          background: linear-gradient(0deg, rgba(0,60,120,0.4), rgba(0,120,200,0.1));
          transform: rotateX(-90deg);
          transform-origin: top center;
          box-shadow: 0 10px 40px rgba(0,180,255,0.3);
          z-index: 0;
        }

        .chatbox--active {
          transform: translateY(0) scale(1) rotateX(2deg) rotateY(0deg) !important;
          opacity: 1 !important;
          pointer-events: auto !important;
        }
        .chatbox--active:hover {
          transform: translateY(0) scale(1) rotateX(0deg) rotateY(0deg) !important;
        }

        /* ================= INNER PANEL ================= */
        .chatbox__panel {
          position: relative;
          width: 100%;
          height: 100%;
          background: linear-gradient(160deg, #050f20 0%, #020810 100%);
          border: 1.5px solid rgba(0,200,255,0.5);
          border-radius: 14px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
          box-shadow:
            0 0 0 1px rgba(0,180,255,0.08),
            inset 0 1px 0 rgba(255,255,255,0.07),
            inset 0 0 60px rgba(0,180,255,0.03),
            0 0 40px rgba(0,180,255,0.15),
            0 30px 80px rgba(0,0,0,0.8);
        }

        /* Glowing top scan line */
        .chatbox__panel::before {
          content: '';
          position: absolute;
          top: 0; left: 30px; right: 30px; height: 2px;
          background: linear-gradient(90deg, transparent, #00d4ff 30%, #00eeff 50%, #ff4ecd 75%, transparent);
          opacity: 0.9;
          z-index: 20;
          box-shadow: 0 0 16px rgba(0,212,255,0.9), 0 0 40px rgba(0,212,255,0.4);
        }

        /* Grid overlay */
        .chatbox__panel::after {
          content: '';
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(0,180,255,0.022) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,180,255,0.022) 1px, transparent 1px);
          background-size: 38px 38px;
          pointer-events: none;
          z-index: 0;
        }

        /* Glowing side edges */
        .edge-l, .edge-r {
          position: absolute; top: 14px; bottom: 14px; width: 2px; z-index: 10;
        }
        .edge-l {
          left: 0;
          background: linear-gradient(180deg, transparent, rgba(0,212,255,0.7) 30%, rgba(0,212,255,0.7) 70%, transparent);
          box-shadow: 0 0 12px rgba(0,212,255,0.6);
        }
        .edge-r {
          right: 0;
          background: linear-gradient(180deg, transparent, rgba(0,212,255,0.7) 30%, rgba(0,212,255,0.7) 70%, transparent);
          box-shadow: 0 0 12px rgba(0,212,255,0.6);
        }
        .edge-b {
          position: absolute; bottom: 0; left: 14px; right: 14px; height: 2px; z-index: 10;
          background: linear-gradient(90deg, transparent, rgba(0,212,255,0.6) 30%, rgba(0,212,255,0.6) 70%, transparent);
          box-shadow: 0 0 10px rgba(0,212,255,0.5);
        }

        /* Corner brackets */
        .cor { position: absolute; width: 26px; height: 26px; z-index: 15; }
        .cor svg { width: 26px; height: 26px; }
        .cor-tl { top: 0; left: 0; }
        .cor-tr { top: 0; right: 0; }
        .cor-bl { bottom: 0; left: 0; }
        .cor-br { bottom: 0; right: 0; }

        /* Right tick marks */
        .ticks {
          position: absolute; right: 5px; top: 50%; transform: translateY(-50%);
          display: flex; flex-direction: column; gap: 5px; z-index: 10;
        }
        .tk { width: 6px; height: 1px; background: rgba(0,200,255,0.4); }
        .tk.b { width: 9px; background: rgba(0,200,255,0.75); }

        /* ================= HEADER ================= */
        .chatbox__header {
          position: relative; z-index: 5;
          display: flex; align-items: center;
          height: 82px; padding: 0 16px 0 0;
          border-bottom: 1px solid rgba(0,180,255,0.18);
          background: linear-gradient(180deg, rgba(0,25,55,0.7) 0%, rgba(0,12,30,0.4) 100%);
          flex-shrink: 0;
        }

        /* Avatar */
        .chatbox__image--header {
          width: 82px; height: 82px; flex-shrink: 0;
          position: relative; display: flex; align-items: center; justify-content: center;
        }
        .av-r1 {
          position: absolute; inset: 2px; border-radius: 50%;
          border: 1.5px solid rgba(0,210,255,0.55);
          box-shadow: 0 0 18px rgba(0,210,255,0.5), inset 0 0 14px rgba(0,210,255,0.08);
          animation: rp 2.5s ease-in-out infinite;
        }
        .av-r2 {
          position: absolute; inset: 0; border-radius: 50%;
          border: 1px solid rgba(0,210,255,0.2);
          animation: rp 2.5s ease-in-out infinite reverse;
        }
        @keyframes rp { 0%,100%{opacity:.7;} 50%{opacity:.25;} }
        .av-N {
          width: 52px; height: 52px; border-radius: 50%;
          background: radial-gradient(circle, #041828, #010c1e);
          border: 1.5px solid rgba(0,210,255,0.65);
          display: flex; align-items: center; justify-content: center;
          font-size: 22px; font-weight: 700; color: #00d4ff;
          position: relative; z-index: 2;
          box-shadow: 0 0 22px rgba(0,212,255,0.55), inset 0 2px 0 rgba(255,255,255,0.1);
        }
        .av-N::after {
          content: ''; position: absolute; inset: 0; border-radius: 50%;
          background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 50%);
          pointer-events: none;
        }

        .chatbox__content--header { flex: 1; padding-left: 4px; }

        .chatbox__heading--header {
          display: flex; align-items: center; gap: 7px;
          font-size: 18px; font-weight: 700; color: #e8f4ff;
          letter-spacing: 2px; margin: 0;
          text-shadow: 0 0 20px rgba(0,212,255,0.4);
        }
        .vai-badge {
          width: 18px; height: 18px; border-radius: 50%;
          background: #00d4ff; display: flex; align-items: center; justify-content: center;
          box-shadow: 0 0 8px rgba(0,212,255,0.7); flex-shrink: 0;
        }
        .vai-badge svg { width: 11px; height: 11px; }

        .chatbox__description--header {
          font-size: 10px; letter-spacing: 2.5px; color: #00d4ff;
          margin: 3px 0 0; opacity: .85;
          display: flex; align-items: center; gap: 5px;
        }

        /* Online dot */
        .vai-online {
          display: inline-block; width: 6px; height: 6px; border-radius: 50%;
          background: #1aff90; box-shadow: 0 0 7px #1aff90;
          animation: vai-pulse 2s infinite;
        }
        @keyframes vai-pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }

        /* Header action buttons */
        .hdr-btns { display: flex; gap: 6px; }
        .hbtn {
          width: 32px; height: 32px;
          border: 1px solid rgba(0,200,255,0.3); border-radius: 7px;
          background: linear-gradient(145deg, rgba(0,40,80,0.6), rgba(0,20,50,0.4));
          color: #00d4ff; display: flex; align-items: center; justify-content: center;
          cursor: pointer; transition: all .2s;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.07), 0 2px 4px rgba(0,0,0,0.4);
        }
        .hbtn:hover {
          background: linear-gradient(145deg, rgba(0,80,150,0.5), rgba(0,50,120,0.4));
          box-shadow: 0 0 12px rgba(0,200,255,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        }
        .hbtn.x { border-color: rgba(255,80,200,0.4); color: #ff4ecd; }
        .hbtn.x:hover { box-shadow: 0 0 12px rgba(255,80,200,0.3); }

        /* Dot row */
        .dot-row {
          position: relative; z-index: 4;
          display: flex; gap: 5px; justify-content: center;
          padding: 5px 0;
          border-bottom: 1px solid rgba(0,200,255,0.07);
          flex-shrink: 0;
        }
        .dot-row span { width: 5px; height: 5px; border-radius: 50%; background: rgba(0,200,255,0.22); }
        .dot-row span.lit { background: #00d4ff; box-shadow: 0 0 6px #00d4ff; }

        /* ================= MESSAGES ================= */
        .chatbox__messages {
          position: relative; z-index: 1;
          padding: 14px; flex: 1;
          overflow-y: auto;
          display: flex; flex-direction: column-reverse;
          gap: 10px;
          background:
            radial-gradient(ellipse at top left, rgba(0,212,255,0.06) 0%, transparent 55%),
            radial-gradient(ellipse at bottom right, rgba(123,94,167,0.07) 0%, transparent 50%),
            rgba(4, 8, 20, 0.85);
          scrollbar-width: thin;
          scrollbar-color: rgba(0,212,255,0.2) transparent;
        }

        /* Welcome screen shown when no messages */
        .vai-welcome {
          display: flex; flex-direction: column; align-items: center;
          gap: 12px; text-align: center;
          position: absolute; inset: 0;
          justify-content: center;
          pointer-events: none;
        }
        .vai-welcome.hidden { display: none; }

        .orb-wrap {
          width: 130px; height: 130px; position: relative;
          display: flex; align-items: center; justify-content: center;
          margin-bottom: 4px;
        }
        .orb-ring {
          position: absolute; inset: 0; border-radius: 50%;
          border: 1px solid rgba(0,200,255,0.2);
          animation: spin 14s linear infinite;
        }
        .orb-ring.r2 {
          inset: 14px; border-color: rgba(0,200,255,0.12);
          animation: spin 20s linear infinite reverse;
        }
        .orb-dot {
          position: absolute; top: 50%; left: 50%;
          width: 6px; height: 6px; border-radius: 50%;
          background: #00d4ff; box-shadow: 0 0 10px #00d4ff;
          animation: dotorb 14s linear infinite;
        }
        .orb-dot.d2 { animation: dotorb2 20s linear infinite reverse; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes dotorb {
          0%   { transform: translate(-50%,-50%) rotate(0deg)   translateX(63px) rotate(0deg); }
          100% { transform: translate(-50%,-50%) rotate(360deg) translateX(63px) rotate(-360deg); }
        }
        @keyframes dotorb2 {
          0%   { transform: translate(-50%,-50%) rotate(0deg)   translateX(48px) rotate(0deg); }
          100% { transform: translate(-50%,-50%) rotate(360deg) translateX(48px) rotate(-360deg); }
        }

        .big-N {
          width: 58px; height: 58px; border-radius: 50%;
          background: radial-gradient(circle at 35% 35%, #063050, #010e22);
          border: 1.5px solid rgba(0,200,255,0.5);
          display: flex; align-items: center; justify-content: center;
          font-size: 26px; font-weight: 700; color: #00d4ff;
          box-shadow: 0 0 30px rgba(0,212,255,0.6), inset 0 0 20px rgba(0,212,255,0.1), inset 0 2px 0 rgba(255,255,255,0.12);
          position: relative; z-index: 2;
          animation: npulse 3s ease-in-out infinite;
        }
        .big-N::after {
          content: ''; position: absolute; inset: 0; border-radius: 50%;
          background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 55%);
          pointer-events: none;
        }
        @keyframes npulse {
          0%,100% { box-shadow: 0 0 22px rgba(0,212,255,0.5); }
          50%      { box-shadow: 0 0 48px rgba(0,212,255,0.8); }
        }

        .w-title {
          font-size: 18px; font-weight: 700; letter-spacing: 2.5px;
          color: #e8f4ff; text-shadow: 0 0 24px rgba(0,212,255,0.35);
        }
        .w-sub { font-size: 10px; letter-spacing: 1.8px; color: #4a7090; line-height: 2; }

        /* Bot bubble */
        .messages__item {
          max-width: 72%;
          padding: 10px 14px;
          border-radius: 16px 16px 16px 4px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(0,212,255,0.12);
          color: var(--text-main);
          align-self: flex-start;
          word-wrap: break-word;
          white-space: pre-wrap;
          font-size: 13px;
          line-height: 1.55;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }

        /* User bubble */
        .messages__item--operator {
          border-radius: 16px 16px 4px 16px;
          background: linear-gradient(135deg, rgba(0,120,180,0.3), rgba(123,94,167,0.25));
          border-color: rgba(0,212,255,0.25);
          color: var(--text-main);
          align-self: flex-end;
          box-shadow: 0 0 14px rgba(0,212,255,0.1);
        }

        /* Link inside bubble */
        a.vai-link {
          color: var(--primary); text-decoration: underline; word-break: break-all;
        }

        /* ================= FOOTER / INPUT BAR ================= */
        .chatbox__footer {
          position: relative; z-index: 2;
          padding: 12px 14px;
          background: rgba(2,8,20,0.75);
          display: flex; gap: 0; align-items: center;
          border-top: 1px solid var(--border-subtle);
          flex-shrink: 0;
        }

        .chatbox__footer-inner {
          display: flex; align-items: center; width: 100%;
          border: 1.5px solid rgba(0,200,255,0.5);
          border-radius: 10px;
          background: linear-gradient(180deg, rgba(0,20,45,0.9), rgba(0,10,25,0.95));
          box-shadow: 0 0 20px rgba(0,200,255,0.18), inset 0 1px 0 rgba(255,255,255,0.06);
          overflow: hidden;
        }

        .chatbox__footer input {
          flex: 1;
          padding: 0 14px;
          height: 50px;
          background: transparent;
          border: none;
          color: var(--text-main);
          font-size: 13px;
          outline: none;
          font-family: inherit;
        }
        .chatbox__footer input::placeholder { color: var(--text-dim); }

        /* Send button — octagon 3D */
        .chatbox__footer button {
          width: 52px; height: 50px; flex-shrink: 0;
          border: none;
          border-left: 1px solid rgba(0,200,255,0.2);
          background: linear-gradient(145deg, rgba(0,180,255,0.32), rgba(0,100,200,0.22));
          color: var(--primary);
          display: flex; align-items: center; justify-content: center;
          cursor: pointer;
          clip-path: polygon(20% 0%,80% 0%,100% 20%,100% 80%,80% 100%,20% 100%,0% 80%,0% 20%);
          transition: background .2s, box-shadow .2s;
          position: relative;
        }
        .chatbox__footer button::before {
          content: ''; position: absolute; inset: 3px;
          clip-path: polygon(20% 0%,80% 0%,100% 20%,100% 80%,80% 100%,20% 100%,0% 80%,0% 20%);
          background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 50%);
          pointer-events: none;
        }
        .chatbox__footer button:hover {
          background: linear-gradient(145deg, rgba(0,220,255,0.45), rgba(0,140,230,0.35));
          box-shadow: 0 0 28px rgba(0,212,255,0.5);
        }
        .chatbox__footer button svg {
          width: 15px; height: 15px;
          fill: none; stroke: #00d4ff;
          stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
        }

        /* ================= STATUS ================= */
        .status {
          padding: 6px 12px;
          color: var(--text-dim);
          font-size: 11px; text-align: center; letter-spacing: 0.3px;
          position: relative; z-index: 2;
          background: rgba(2,8,20,0.7);
          flex-shrink: 0;
        }

        /* ================= STATUS BAR ================= */
        .chatbox__statusbar {
          position: relative; z-index: 5;
          padding: 5px 16px 7px;
          background: rgba(1,5,14,0.96);
          border-top: 1px solid rgba(0,200,255,0.1);
          display: flex; align-items: center; justify-content: center;
          flex-shrink: 0;
        }
        .stat-seg {
          display: flex; align-items: center; gap: 5px;
          font-size: 9px; letter-spacing: 1.8px; color: #2a4560; padding: 0 14px;
        }
        .stat-div { width: 1px; height: 10px; background: rgba(0,200,255,0.15); }
        .lock-ic { width: 9px; height: 9px; stroke: #2a4560; fill: none; stroke-width: 1.5; }
        .prog-l {
          position: absolute; left: 14px; top: 50%; transform: translateY(-50%);
          display: flex; gap: 2px; align-items: center;
        }
        .prog-r {
          position: absolute; right: 14px; top: 50%; transform: translateY(-50%);
          display: flex; gap: 2px; align-items: center;
        }
        .ps { height: 3px; border-radius: 2px; background: rgba(0,200,255,0.22); }
        .ps.w20{width:20px;} .ps.w12{width:12px;} .ps.w6{width:6px;background:rgba(0,200,255,0.1);}
        .sdots {
          position: absolute; bottom: 3px; left: 50%; transform: translateX(-50%);
          display: flex; gap: 4px;
        }
        .sdot { width: 4px; height: 4px; border-radius: 50%; background: rgba(0,200,255,0.18); }
        .sdot.on { background: #00d4ff; box-shadow: 0 0 5px #00d4ff; }

        /* ================= COPYRIGHT ================= */
        .chatbox__copyright {
          position: relative; z-index: 5;
          background: rgba(1,4,12,0.97);
          border-top: 1px solid rgba(0,212,255,0.08);
          color: var(--text-dim);
          font-size: 9px; text-align: center;
          padding: 5px 0; letter-spacing: 1px;
          flex-shrink: 0;
        }
        .chatbox__copyright a {
          color: #1e3a55; text-decoration: none; transition: color .2s;
        }
        .chatbox__copyright a:hover { color: var(--primary); }

        /* ================= TOGGLE BUTTON ICON ================= */
        .vai-chatbox .chatbox__button i { font-size: 26px; line-height: 1; }
      `;

      const styleEl = document.createElement('style');
      styleEl.setAttribute('type','text/css');
      styleEl.appendChild(document.createTextNode(css));
      shadow.appendChild(styleEl);

      // =====================================================
      // 3D NOAH UI — HTML (merged with your widget structure)
      // =====================================================
      const container = document.createElement('div');
      container.className = 'vai-chatbox';
      container.innerHTML = `
        <div class="chatbox__support" id="vai_support" aria-hidden="true">

          <!-- 3D INNER PANEL WRAPPER -->
          <div class="chatbox__panel">

            <!-- Glowing side/bottom edges -->
            <div class="edge-l"></div>
            <div class="edge-r"></div>
            <div class="edge-b"></div>

            <!-- Corner L-brackets -->
            <div class="cor cor-tl">
              <svg viewBox="0 0 26 26" fill="none"><path d="M26 2H5a3 3 0 0 0-3 3v21" stroke="#00d4ff" stroke-width="2" opacity="0.95"/></svg>
            </div>
            <div class="cor cor-tr">
              <svg viewBox="0 0 26 26" fill="none"><path d="M0 2h21a3 3 0 0 1 3 3v21" stroke="#00d4ff" stroke-width="2" opacity="0.95"/></svg>
            </div>
            <div class="cor cor-bl">
              <svg viewBox="0 0 26 26" fill="none"><path d="M26 24H5a3 3 0 0 1-3-3V0" stroke="#00d4ff" stroke-width="2" opacity="0.95"/></svg>
            </div>
            <div class="cor cor-br">
              <svg viewBox="0 0 26 26" fill="none"><path d="M0 24h21a3 3 0 0 0 3-3V0" stroke="#00d4ff" stroke-width="2" opacity="0.95"/></svg>
            </div>

            <!-- Right tick marks -->
            <div class="ticks">
              <div class="tk b"></div><div class="tk"></div><div class="tk"></div>
              <div class="tk b"></div><div class="tk"></div><div class="tk"></div>
              <div class="tk b"></div>
            </div>

            <!-- HEADER -->
            <div class="chatbox__header">
              <div class="chatbox__image--header">
                <div class="av-r2"></div>
                <div class="av-r1"></div>
                <div class="av-N">N</div>
              </div>
              <div class="chatbox__content--header">
                <h4 class="chatbox__heading--header">
                  NOAH
                  <div class="vai-badge">
                    <svg viewBox="0 0 11 11" fill="none">
                      <path d="M2 5.5l2.5 2.5 4.5-4.5" stroke="#020d1a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </div>
                </h4>
                <p class="chatbox__description--header">
                  <span class="vai-online"></span>
                  INSTITUTIONAL LANGUAGE MODEL
                </p>
              </div>
              <div class="hdr-btns">
                <div class="hbtn" title="Timer">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2"/><path d="M9 3h6"/>
                  </svg>
                </div>
                <div class="hbtn" title="Expand">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
                  </svg>
                </div>
                <div class="hbtn x" id="vai_close_btn" title="Close">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 6L6 18M6 6l12 12"/>
                  </svg>
                </div>
              </div>
            </div>

            <!-- Dot row -->
            <div class="dot-row">
              <span></span><span class="lit"></span><span></span><span></span>
            </div>

            <!-- MESSAGES -->
            <div class="chatbox__messages" id="vai_messages" aria-live="polite">
              <!-- Welcome screen -->
              <div class="vai-welcome" id="vai_welcome">
                <div class="orb-wrap">
                  <div class="orb-ring"></div>
                  <div class="orb-ring r2"></div>
                  <div class="orb-dot"></div>
                  <div class="orb-dot d2"></div>
                  <div class="big-N">N</div>
                </div>
                <div class="w-title">HI, I'M NOAH.</div>
                <div class="w-sub">YOUR INSTITUTIONAL AI ASSISTANT.<br>HOW CAN I HELP YOU TODAY?</div>
              </div>
            </div>

            <!-- STATUS (typing / error feedback) -->
            <div class="status" id="vai_status" style="display:none"></div>

            <!-- FOOTER / INPUT BAR -->
            <div class="chatbox__footer">
              <div class="chatbox__footer-inner">
                <input id="vai_input" type="text" placeholder="Write a message..." aria-label="Message" />
                <button id="vai_send" type="button" aria-label="Send">
                  <svg viewBox="0 0 24 24">
                    <line x1="22" y1="2" x2="11" y2="13"/>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                  </svg>
                </button>
              </div>
            </div>

            <!-- STATUS BAR -->
            <div class="chatbox__statusbar">
              <div class="prog-l">
                <div class="ps w20"></div><div class="ps w12"></div><div class="ps w6"></div>
                <div style="width:4px;height:4px;border-radius:50%;background:rgba(0,200,255,0.28);margin-left:2px;"></div>
                <div style="width:4px;height:4px;border-radius:50%;background:rgba(0,200,255,0.16);"></div>
              </div>
              <div class="stat-seg">
                <svg class="lock-ic" viewBox="0 0 24 24">
                  <rect x="3" y="11" width="18" height="11" rx="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
                SECURE CONNECTION
              </div>
              <div class="stat-div"></div>
              <div class="stat-seg">END-TO-END ENCRYPTED</div>
              <div class="prog-r">
                <div class="ps w6"></div><div class="ps w12"></div><div class="ps w20"></div>
              </div>
              <div class="sdots">
                <div class="sdot on"></div><div class="sdot"></div><div class="sdot"></div>
              </div>
            </div>

            <!-- COPYRIGHT -->
            <div class="chatbox__copyright">
              &copy; 2026 <a href="https://www.cogniaistudios.com" target="_blank" rel="noopener noreferrer">CogniAI Studios</a>. All rights reserved.
            </div>

          </div><!-- /chatbox__panel -->
        </div><!-- /chatbox__support -->

        <!-- FLOATING TOGGLE BUTTON (original, kept hidden — replaced by external) -->
        <div class="chatbox__button">
          <button id="vai_toggle" title="Chat with Noah" style="display:none">
            <i class='bx bxs-message'></i>
          </button>
        </div>
      `;

      shadow.appendChild(container);

      // =====================================================
      // EXTERNAL FLOATING TOGGLE BUTTON (your original code)
      // =====================================================
      const originalToggleBtn = shadow.getElementById('vai_toggle');
      if (originalToggleBtn) originalToggleBtn.style.display = 'none';

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
        background: '#17f38c',
        color: '#fff',
        cursor: 'pointer',
        zIndex: '2147483001',
        fontSize: '22px',
        outline: 'none'
      });
      document.body.appendChild(externalToggle);

      // =====================================================
      // JS — ALL YOUR ORIGINAL LOGIC, UNTOUCHED
      // =====================================================
      const support    = shadow.getElementById('vai_support');
      const toggle     = shadow.getElementById('vai_toggle');
      const messagesEl = shadow.getElementById('vai_messages');
      const inputEl    = shadow.getElementById('vai_input');
      const sendBtn    = shadow.getElementById('vai_send');
      const statusEl   = shadow.getElementById('vai_status');
      const welcomeEl  = shadow.getElementById('vai_welcome');
      const closeBtn   = shadow.getElementById('vai_close_btn');

      function setStatus(msg, show=true) {
        if (!msg) {
          statusEl.style.display = 'none';
          statusEl.textContent = '';
          return;
        }
        statusEl.style.display = show ? 'block' : 'none';
        statusEl.textContent = msg;
      }

      // Toggle open/close
      function toggleChat() {
        support.classList.toggle('chatbox--active');
        const isOpen = support.classList.contains('chatbox--active');
        support.setAttribute('aria-hidden', String(!isOpen));
        if (isOpen) inputEl.focus();
      }

      if (toggle) {
        try { toggle.addEventListener('click', toggleChat); } catch(e) {}
      }
      externalToggle.addEventListener('click', toggleChat);
      if (closeBtn) closeBtn.addEventListener('click', toggleChat);

      // Hide welcome screen once first message sent
      function hideWelcome() {
        if (welcomeEl && !welcomeEl.classList.contains('hidden')) {
          welcomeEl.classList.add('hidden');
        }
      }

      function appendMessage(text, who='veronica') {
        hideWelcome();
        const div = document.createElement('div');
        div.className = 'messages__item ' + (who === 'you' ? 'messages__item--operator' : 'messages__item--visitor');
        div.textContent = text;
        messagesEl.insertBefore(div, messagesEl.firstChild);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
      }

      function appendHtmlMessage(htmlContent, who='veronica') {
        hideWelcome();
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
        appendMessage(val, 'you');
        inputEl.value = '';
        setStatus('Sending...');
        try {
          const typingDiv = appendMessage('Just a moment...', 'veronica');

          const doFetch = async () => {
            const payload = {
              message: val,
              session_id: SESSION_ID,
              url: location.href,
              user_agent: navigator.userAgent || '',
              language: navigator.language || '',
              timestamp: (new Date()).toISOString()
            };
            console.log('[VERONICA] POST', BASE_API + '/predict', 'payload:', payload);
            return await fetch(BASE_API + '/predict', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(payload),
              keepalive: true
            });
          };

          let res = await doFetch();

          if (typingDiv && typingDiv.parentNode) messagesEl.removeChild(typingDiv);

          if (!res.ok) {
            console.warn('[VERONICA] first fetch not ok, status=', res.status);
            setStatus('Veronica is waking up — retrying in a few seconds...', true);
            await new Promise(r => setTimeout(r, 3500));
            res = await doFetch();
            if (!res.ok) {
              console.error('[VERONICA] retry failed, status=', res.status);
              setStatus(`Server error (${res.status}). Try again later.`);
              return;
            }
          }

          let data;
          try { data = await res.json(); }
          catch (e) {
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

      // Expose public API
      window.VERONICA_WIDGET = {
        baseUrl: BASE_API,
        sessionId: SESSION_ID,
        open:  () => { support.classList.add('chatbox--active');    support.setAttribute('aria-hidden','false'); },
        close: () => { support.classList.remove('chatbox--active'); support.setAttribute('aria-hidden','true');  },
        send:  (msg) => { inputEl.value = msg; sendMessage(); }
      };

      console.log('[VERONICA WIDGET] ready (shadow DOM + 3D NOAH UI)');

    } catch (e) {
      console.error('[VERONICA WIDGET] init failed', e);
    }
  });

})();
