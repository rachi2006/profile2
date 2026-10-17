"""Lightweight prep for photos that already have a plain white background (no rembg needed).
crop to a square head+shoulders, mask the white background, CLAHE for local contrast."""
import sys, cv2, numpy as np
from PIL import Image
inp, out = sys.argv[1], sys.argv[2]
img = cv2.imread(inp)
h, w = img.shape[:2]
side = int(w*0.82); top = int(h*0.03); x0 = int(w*0.5 - side*0.5) + 8
crop = img[top:top+side, x0:x0+side]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# background mask: near-white connected to border
bg = (gray > 236).astype(np.uint8)
num, lab = cv2.connectedComponents(bg)
border = set(lab[0,:])|set(lab[-1,:])|set(lab[:,0])|set(lab[:,-1])
bgmask = np.isin(lab, [l for l in border if l!=0]).astype(np.float32)
fg = 1-cv2.GaussianBlur(bgmask,(0,0),1.5)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
g = clahe.apply(gray).astype(np.float32)
# normalise the SUBJECT only: stretch percentiles, then lift the median so skin
# lands in the sparse end of the ascii ramp (dark skin tones otherwise go blank/blobby)
m = fg > 0.5
lo, hi = np.percentile(g[m], [1, 99])
g = np.clip((g-lo)/(hi-lo), 0, 1)
med = np.median(g[m]); target = 0.66
g = np.power(g, np.log(target)/np.log(max(med,1e-3)))
g = (g*255).astype(np.uint8)
o = g.astype(np.float32)*fg + 255*(1-fg)
Image.fromarray(np.clip(o,0,255).astype(np.uint8), mode="L").save(out)
print("ok", o.shape)
