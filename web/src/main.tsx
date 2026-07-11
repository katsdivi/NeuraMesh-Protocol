import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './styles/app.css';

// Service workers only exist in secure contexts (https / localhost).
// Over plain http on the trusted LAN the register call would throw —
// the app is fully functional without one (it's just the offline shell
// cache for TLS-fronted deployments).
if ('serviceWorker' in navigator && window.isSecureContext) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      /* not fatal — live app works without the shell cache */
    });
  });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
