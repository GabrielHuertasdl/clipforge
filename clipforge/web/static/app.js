// Keyboard-first review: A approve, R reject, E caption, arrows navigate, space play/pause.
// Number keys pick a rejection reason once the reason row is open.

function panel() { return document.querySelector('.reviewer'); }
function player() { return document.getElementById('player'); }

function toggleReasons(force) {
  const box = document.getElementById('reasons');
  if (!box) return;
  box.hidden = force === undefined ? !box.hidden : !force;
  if (!box.hidden) {
    const first = box.querySelector('button');
    if (first) first.focus();
  }
}

function go(uid) {
  if (!uid) return;
  htmx.ajax('GET', '/panel/' + uid, { target: '#panel', swap: 'innerHTML' });
}

function isTyping(el) {
  return el && (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT');
}

document.addEventListener('keydown', (event) => {
  const box = panel();
  if (!box) return;

  // While typing, only Escape (blur) and Ctrl/Cmd+Enter (approve) are handled.
  if (isTyping(document.activeElement)) {
    if (event.key === 'Escape') { document.activeElement.blur(); }
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      document.getElementById('approve-form').requestSubmit();
      event.preventDefault();
    }
    return;
  }

  const key = event.key.toLowerCase();
  const reasons = document.getElementById('reasons');

  if (reasons && !reasons.hidden && /^[1-9]$/.test(key)) {
    const buttons = reasons.querySelectorAll('button');
    const chosen = buttons[parseInt(key, 10) - 1];
    if (chosen) { chosen.click(); event.preventDefault(); }
    return;
  }

  switch (key) {
    case 'a':
      document.getElementById('approve-form').requestSubmit();
      event.preventDefault();
      break;
    case 'r':
      toggleReasons();
      event.preventDefault();
      break;
    case 'e': {
      const caption = document.getElementById('caption');
      if (caption) { caption.focus(); caption.select(); }
      event.preventDefault();
      break;
    }
    case 'arrowleft':
      go(box.dataset.prev);
      event.preventDefault();
      break;
    case 'arrowright':
      go(box.dataset.next);
      event.preventDefault();
      break;
    case ' ': {
      const video = player();
      if (video) { video.paused ? video.play() : video.pause(); }
      event.preventDefault();
      break;
    }
    case 'escape':
      toggleReasons(false);
      break;
  }
});

// After every HTMX swap the video element is new: make sure it plays.
document.body.addEventListener('htmx:afterSwap', () => {
  const video = player();
  if (video && video.hasAttribute('autoplay')) {
    video.play().catch(() => {});   // some mobile browsers refuse without a gesture
  }
});
