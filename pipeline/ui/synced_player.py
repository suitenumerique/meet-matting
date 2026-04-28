"""
Synchronized three-video player rendered as an HTML component.
A single Play / Pause button controls all three videos simultaneously.
"""

import base64
from pathlib import Path

import streamlit.components.v1 as components

_LABELS: dict[str, str] = {
    "original":  "Original",
    "composite": "Composite",
    "mask":      "Mask",
}


def display_synced_player(paths: dict[str, Path], height: int = 420) -> None:
    """Render Original / Composite / Mask videos with a shared Play / Pause button.

    Args:
        paths:  Dict with keys 'original', 'composite', 'mask' mapping to Path objects.
                Missing or nonexistent paths are skipped gracefully.
        height: Height of the iframe component in pixels.
    """
    video_divs: list[str] = []

    for key, label in _LABELS.items():
        path = paths.get(key)
        if path is None or not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        video_divs.append(
            f'<div style="flex:1;min-width:0;">'
            f'<p style="margin:0 0 4px;font-size:12px;color:#888;">{label}</p>'
            f'<video class="sv" src="data:video/mp4;base64,{b64}"'
            f' style="width:100%;border-radius:4px;" controls preload="auto"></video>'
            f'</div>'
        )

    if not video_divs:
        return

    html = """<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;font-family:sans-serif;">
<div style="display:flex;gap:8px;margin-bottom:10px;">
"""
    html += "\n".join(video_divs)
    html += """
</div>
<div style="display:flex;gap:8px;">
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
</script>
</body>
</html>"""

    components.html(html, height=height)
