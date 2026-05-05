"""
Synchronized multi-video player rendered as an HTML component.
A single Play / Pause button controls all videos simultaneously.

The iframe auto-resizes to fit its content via the Streamlit postMessage API
so that the Play/Pause controls are always visible regardless of video orientation
(landscape or portrait).
"""

import base64
from pathlib import Path

import streamlit.components.v1 as components

_LABELS: dict[str, str] = {
    "original": "Original",
    "mask": "Masque alpha (apres post-process)",
    "raw": "Composite brut (sans post-process)",
    "composite": "Composite final (apres post-process)",
}


def display_synced_player(paths: dict[str, Path], height: int = 1100) -> None:
    """Render synchronized videos with a shared Play / Pause button.

    The iframe sends its real content height to Streamlit after the DOM is ready
    so the component auto-expands for portrait videos without cutting off controls.

    Args:
        paths:  Dict with keys from _LABELS mapping to Path objects.
                Missing or non-existent paths are silently skipped.
        height: Initial iframe height in pixels (Streamlit allocates this on first
                render; the JS postMessage then adjusts to the real content height).
    """
    video_divs: list[str] = []

    for key, label in _LABELS.items():
        path = paths.get(key)
        if path is None or not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        video_divs.append(
            f'<div style="flex:1;min-width:0;">'
            f'<p style="margin:0 0 4px;font-size:12px;color:#aaa;">{label}</p>'
            f'<video class="sv" src="data:video/mp4;base64,{b64}"'
            f' style="width:100%;max-height:480px;object-fit:contain;'
            f'border-radius:4px;background:#000;" controls preload="auto"></video>'
            f"</div>"
        )

    if not video_divs:
        return

    html = """<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0; font-family: sans-serif; background: transparent; }
  #videos { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }
  #controls { display: flex; gap: 8px; padding-bottom: 4px; }
</style>
</head>
<body>
<div id="videos">
"""
    html += "\n".join(video_divs)
    html += """
</div>
<div id="controls">
  <button id="btnPlay"
    style="padding:6px 20px;cursor:pointer;border-radius:4px;font-size:13px;
           background:#FF4B4B;color:white;border:none;">
    Play all
  </button>
  <button id="btnPause"
    style="padding:6px 20px;cursor:pointer;border-radius:4px;font-size:13px;
           background:white;color:#333;border:1px solid #ccc;">
    Pause
  </button>
</div>
<script>
  function allVideos() {
    return Array.from(document.querySelectorAll('video.sv'));
  }

  document.getElementById('btnPlay').addEventListener('click', function() {
    allVideos().forEach(function(v) {
      v.currentTime = 0;
      var p = v.play();
      if (p !== undefined) { p.catch(function(e) { console.warn('play() blocked:', e); }); }
    });
  });

  document.getElementById('btnPause').addEventListener('click', function() {
    allVideos().forEach(function(v) { v.pause(); });
  });

  // Auto-resize the Streamlit iframe to fit the actual content height.
  // Fires multiple times to catch layout shifts from video metadata loading.
  function sendHeight() {
    var h = document.body.scrollHeight + 8;
    window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: h }, '*');
  }
  document.addEventListener('DOMContentLoaded', sendHeight);
  window.addEventListener('load', sendHeight);
  setTimeout(sendHeight, 200);
  setTimeout(sendHeight, 800);

  // Also resize when any video loads its metadata (actual dimensions become known).
  allVideos().forEach(function(v) {
    v.addEventListener('loadedmetadata', sendHeight);
  });
</script>
</body>
</html>"""

    components.html(html, height=height, scrolling=False)
