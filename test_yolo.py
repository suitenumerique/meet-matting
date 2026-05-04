import time

import numpy as np

from pipeline.core.detector import YoloDetector

print("Initializing YOLO...")
yolo = YoloDetector(model_size="n")
yolo.load()
print("YOLO Loaded.")

print("Creating dummy frame...")
frame = np.zeros((720, 1280, 3), dtype=np.uint8)

print("Running detect...")
t0 = time.time()
boxes = yolo.detect(frame)
t1 = time.time()

print(f"Detected {len(boxes)} boxes in {t1 - t0:.3f}s")
print("Done.")
