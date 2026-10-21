import csv
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import ExifTags, Image

np.random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, FIG, RES = (os.path.join(ROOT, d) for d in ("data", "figures", "results"))
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def save_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


class Tee:  # вивід і в консоль, і в results/
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


sys.stdout = Tee(sys.__stdout__, open(os.path.join(RES, "output.txt"), "w", encoding="utf-8"))


def read(name, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(os.path.join(DATA, name), flags)
    assert img is not None, f"немає файлу data/{name}"
    return img


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def iou(pred, gt):
    pred, gt = pred > 0, gt > 0
    return (pred & gt).sum() / (pred | gt).sum()


def accuracy(pred, gt):
    return np.mean((pred > 0) == (gt > 0))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===================== ЧАСТИНА A: подання зображення =====================
print("=== A. Подання зображення ===")
img = read("photo.jpg")
print("shape:", img.shape, "dtype:", img.dtype, "min:", img.min(), "max:", img.max())
save_csv("a_properties.csv", ["параметр", "значення"],
         [["shape", str(img.shape)], ["dtype", img.dtype], ["min", img.min()], ["max", img.max()],
          ["обсяг, МБ", round(img.nbytes / 2 ** 20, 2)]])

# крок 2: порядок каналів
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(img)
ax[0].set_title("без перетворення (R і B переставлені)")
ax[1].imshow(rgb(img))
ax[1].set_title("після BGR -> RGB")
for a in ax:
    a.set_xlabel("x, піксель"); a.set_ylabel("y, піксель")
save(fig, "a_bgr_rgb.png")

# крок 3: переповнення uint8
wrong = img + 50
right = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
overflow = (wrong < img).any(axis=2)
g0, g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(wrong, cv2.COLOR_BGR2GRAY)
bright_to_dark = (g0 >= 200) & (g1 < 100)
print(f"переповнення: {overflow.mean() * 100:.2f} %, яскраві -> темні: {bright_to_dark.mean() * 100:.2f} %")
save_csv("a_overflow.csv", ["метод", "пікселів з переповненням, %", "яскраві->темні, %"],
         [["img + 50", round(overflow.mean() * 100, 2), round(bright_to_dark.mean() * 100, 2)],
          ["int16 + clip", 0, 0]])

score = cv2.boxFilter(bright_to_dark.astype(np.float32), -1, (256, 256))
cy, cx = np.unravel_index(score.argmax(), score.shape)
y0 = min(max(cy - 128, 0), img.shape[0] - 256)
x0 = min(max(cx - 128, 0), img.shape[1] - 256)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
for a, im, t in zip(ax, (img, wrong, right), ("оригінал", "img + 50 (переповнення)", "int16 + clip")):
    a.imshow(rgb(im[y0:y0 + 256, x0:x0 + 256]))
    a.set_title(t); a.set_xlabel("x, піксель")
ax[0].set_ylabel("y, піксель")
save(fig, "a_overflow.png")

# крок 4: стиснення
print("\nформат  якість  розмір_КБ  PSNR_дБ")
_, png = cv2.imencode(".png", img)
print(f"PNG     -       {len(png) / 1024:9.1f}  inf")
sizes, psnrs = [], []
comp = [["PNG", "-", round(len(png) / 1024, 1), "inf"]]
for q in (95, 75, 50, 25, 10):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    p = psnr(img, cv2.imdecode(buf, cv2.IMREAD_COLOR))
    sizes.append(len(buf) / 1024); psnrs.append(p)
    print(f"JPEG    {q:<6}  {sizes[-1]:9.1f}  {p:.2f}")
    comp.append(["JPEG", q, round(sizes[-1], 1), round(p, 2)])
save_csv("a_compression.csv", ["формат", "якість", "розмір, КБ", "PSNR, дБ"], comp)
fig = plt.figure(figsize=(7, 4.5))
plt.plot(sizes, psnrs, "o-")
for q, s, p in zip((95, 75, 50, 25, 10), sizes, psnrs):
    plt.annotate(f"q={q}", (s, p), xytext=(5, -12), textcoords="offset points")
plt.xscale("log"); plt.grid(alpha=.3)
plt.xlabel("розмір файлу, КБ (логарифмічна шкала)"); plt.ylabel("PSNR, дБ")
save(fig, "a_psnr_size.png")


# крок 5: EXIF
def exif(path):
    e = Image.open(path).getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in e.items()}
    tags.update({ExifTags.TAGS.get(k, k): v for k, v in e.get_ifd(ExifTags.IFD.Exif).items()})
    return tags


print("\nEXIF:")
exif_rows = []
for path in [os.path.join(DATA, n) for n in ("photo.jpg", "object_light1.jpg", "object_light2.jpg")] + \
            sorted(glob.glob(os.path.join(DATA, "dataset", "*.jpg"))):
    t = exif(path)
    exp = f"1/{round(1 / float(t['ExposureTime']))} с" if "ExposureTime" in t else "-"
    print(os.path.basename(path), "| f =", t.get("FocalLength", "-"), "мм | витримка", exp,
          "| ISO", t.get("ISOSpeedRatings", "-"))
    exif_rows.append([os.path.basename(path), t.get("Model", "-"), round(float(t.get("FocalLength", 0)), 2),
                      exp, t.get("ISOSpeedRatings", "-"), t.get("DateTimeOriginal", "-")])
save_csv("a_exif.csv", ["файл", "камера", "фокусна відстань, мм", "витримка", "ISO", "дата"], exif_rows)
print("кадрів у наборі:", len(glob.glob(os.path.join(DATA, "dataset", "*.jpg"))), "(має бути 20)")

# ===================== ЧАСТИНА B: колірні простори =====================
print("\n=== B. Пороги в різних колірних просторах ===")
f1, f2 = read("object_light1.jpg"), read("object_light2.jpg")
m1 = read("object_mask1.png", cv2.IMREAD_GRAYSCALE)
m2 = read("object_mask2.png", cv2.IMREAD_GRAYSCALE)

SPACES = [("RGB", None, [0, 1, 2]),
          ("HSV", cv2.COLOR_BGR2HSV, [0, 1, 2]),
          ("HSV (лише H, S)", cv2.COLOR_BGR2HSV, [0, 1])]
print("простір            межі(нижні)..(верхні)            IoU к1  acc к1  IoU к2  acc к2")
results = {}
b_rows = []
for name, code, chans in SPACES:
    c1 = f1 if code is None else cv2.cvtColor(f1, code)
    c2 = f2 if code is None else cv2.cvtColor(f2, code)
    lo, hi = np.zeros(3, np.uint8), np.full(3, 255, np.uint8)
    for c in chans:
        v = c1[:, :, c][m1 > 0]
        lo[c], hi[c] = np.percentile(v, 1), np.percentile(v, 99)
    p1, p2 = cv2.inRange(c1, lo, hi), cv2.inRange(c2, lo, hi)
    results[name] = (p1, p2)
    print(f"{name:16s} {lo.tolist()}..{hi.tolist()}  {iou(p1, m1):.3f}  {accuracy(p1, m1):.3f}  "
          f"{iou(p2, m2):.3f}  {accuracy(p2, m2):.3f}")
    b_rows.append([name, str(lo.tolist()), str(hi.tolist()), round(iou(p1, m1), 3), round(accuracy(p1, m1), 3),
                   round(iou(p2, m2), 3), round(accuracy(p2, m2), 3)])
print("accuracy порожньої маски (кадр 1, кадр 2):",
      round(accuracy(np.zeros_like(m1), m1), 3), round(accuracy(np.zeros_like(m2), m2), 3))
b_rows.append(["порожня маска", "-", "-", 0, round(accuracy(np.zeros_like(m1), m1), 3), 0,
               round(accuracy(np.zeros_like(m2), m2), 3)])
save_csv("b_thresholds.csv", ["простір", "нижні межі", "верхні межі", "IoU кадр 1", "accuracy кадр 1",
                              "IoU кадр 2", "accuracy кадр 2"], b_rows)

fig, ax = plt.subplots(2, len(SPACES), figsize=(4 * len(SPACES), 8))
for i, (name, _, _) in enumerate(SPACES):
    for r, (pred, gt) in enumerate(zip(results[name], (m1, m2))):
        pic = np.zeros((*gt.shape, 3), np.uint8)
        pic[(pred > 0) & (gt > 0)] = (0, 170, 0)
        pic[(pred > 0) & (gt == 0)] = (230, 30, 30)
        pic[(pred == 0) & (gt > 0)] = (40, 80, 255)
        ax[r, i].imshow(pic)
        ax[r, i].set_title(f"{name}, кадр {r + 1}, IoU = {iou(pred, gt):.2f}", fontsize=9)
        ax[r, i].axis("off")
save(fig, "b_masks.png")

# ===================== ЧАСТИНА C: геометрія та радіометрія =====================
print("\n=== C. Аліасинг, інтерполяція, гамма ===")
# крок 12: зменшення в 4 рази
tex = read("texture.jpg")
size = (tex.shape[1] // 4, tex.shape[0] // 4)
nearest = cv2.resize(tex, size, interpolation=cv2.INTER_NEAREST)
blur_nn = cv2.resize(cv2.GaussianBlur(tex, (0, 0), 2), size, interpolation=cv2.INTER_NEAREST)
area = cv2.resize(tex, size, interpolation=cv2.INTER_AREA)  # еталон без аліасингу
print(f"PSNR відносно INTER_AREA: NEAREST {psnr(nearest, area):.2f} дБ, Gauss+NEAREST {psnr(blur_nn, area):.2f} дБ")
save_csv("c_aliasing.csv", ["метод", "PSNR відносно INTER_AREA, дБ"],
         [["NEAREST", round(psnr(nearest, area), 2)], ["Gauss σ=2 + NEAREST", round(psnr(blur_nn, area), 2)]])
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, t in zip(ax, (area, nearest, blur_nn), ("INTER_AREA (еталон)", "NEAREST", "Gauss σ=2 + NEAREST")):
    a.imshow(rgb(im), interpolation="nearest")
    a.set_title(t); a.set_xlabel("x, піксель"); a.set_ylabel("y, піксель")
save(fig, "c_aliasing.png")

# крок 13: інтерполяція
photo = read("photo.jpg")
H, W = photo.shape[:2]
interp = []
crops = [photo[H // 2 - 256 + dy:H // 2 + 256 + dy, x:x + 512] for dy, x in ((0, 0), (0, W // 2 - 256), (0, W - 512))]
for name, flag in (("NEAREST", cv2.INTER_NEAREST), ("LINEAR", cv2.INTER_LINEAR),
                   ("CUBIC", cv2.INTER_CUBIC), ("LANCZOS4", cv2.INTER_LANCZOS4)):
    ps = [psnr(c, cv2.resize(cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA), (512, 512), interpolation=flag))
          for c in crops]
    print(f"{name:9s} PSNR = {np.mean(ps):.2f} ± {np.std(ps):.2f} дБ")
    interp.append([name] + [round(v, 2) for v in ps] + [round(np.mean(ps), 2), round(np.std(ps), 2)])
save_csv("c_interpolation.csv", ["метод", "фрагмент 1, дБ", "фрагмент 2, дБ", "фрагмент 3, дБ", "середнє, дБ", "std, дБ"], interp)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
small = cv2.resize(crops[0], (64, 64), interpolation=cv2.INTER_AREA)
for a, (name, flag) in zip(ax, (("NEAREST", 0), ("LINEAR", 1), ("CUBIC", 2), ("LANCZOS4", 4))):
    a.imshow(rgb(cv2.resize(small, (512, 512), interpolation=flag)[100:260, 100:260]), interpolation="nearest")
    a.set_title(name); a.set_xlabel("x, піксель")
save(fig, "c_interpolation.png")

# крок 14: гамма-корекція
x = np.arange(256)
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
for a, g in zip(ax[:3], (0.5, 1.0, 2.2)):
    lut = (255 * (x / 255) ** g).astype(np.uint8)
    a.imshow(rgb(cv2.LUT(photo, lut)))
    a.set_title(f"γ = {g}")
    ax[3].plot(x, lut, label=f"γ = {g}")
ax[3].set_xlabel("вхід, рівні 0-255"); ax[3].set_ylabel("вихід, рівні 0-255")
ax[3].legend(); ax[3].grid(alpha=.3)
save(fig, "c_gamma.png")
gam = []
for a_, b_ in ((0, 255), (50, 200)):
    light = ((a_ / 255) ** 2.2 + (b_ / 255) ** 2.2) / 2
    print(f"піксели {a_} і {b_}: середнє кодів = {(a_ + b_) / 2}, правильний код = {255 * light ** (1 / 2.2):.1f}")
    gam.append([a_, b_, (a_ + b_) / 2, round(255 * light ** (1 / 2.2), 1)])
save_csv("c_gamma_average.csv", ["v1", "v2", "середнє кодів", "код середньої освітленості"], gam)