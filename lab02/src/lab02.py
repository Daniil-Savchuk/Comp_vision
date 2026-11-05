import csv
import datetime
import glob
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")  # малюємо у файли, а не у вікна
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ======================================================================
# 0. НАЛАШТУВАННЯ  (змінюйте тільки цей блок)
# ======================================================================
SQUARE_MM = 26.0
PATTERN = (9, 6)
DETECT_MAX_SIDE = 1600
LINE_PTS = ((0, 1800), (6000, 1800))
REPEATS = 5
N_LIST = (5, 8, 10, 12, 15, 20)
OUTLIER_FACTOR = 2.0

np.random.seed(42)
rng = np.random.default_rng(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(ROOT, "data", "calib")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(FIG, "corners"), exist_ok=True)


class Tee:
    """Дублює print() і в консоль, і у файл results/output.txt."""
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


sys.stdout = Tee(sys.__stdout__, open(os.path.join(RES, "output.txt"), "w", encoding="utf-8"))


def save_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


assert SQUARE_MM is not None, "Впишіть SQUARE_MM (розмір клітини в мм, виміряний лінійкою) на початку файлу"

# ======================================================================
# 1. ЗАВАНТАЖЕННЯ КАДРІВ
# ======================================================================
paths = sorted(glob.glob(os.path.join(CALIB, "*.jpg")))
assert paths, f"Немає фото в {CALIB}"
imgs = [cv2.imread(p) for p in paths]
names = [os.path.basename(p) for p in paths]


sizes = [(im.shape[1], im.shape[0]) for im in imgs]
size = max(set(sizes), key=sizes.count)
if any(s != size for s in sizes):
    print("УВАГА: кадри мають різний розмір (напр. частина знята вертикально). Беру лише розмір", size)
keep = [i for i, s in enumerate(sizes) if s == size]
imgs, names = [imgs[i] for i in keep], [names[i] for i in keep]
print(f"Кадрів у data/calib: {len(paths)}, однакового розміру {size[0]}x{size[1]}: {len(imgs)}")

# ======================================================================
# 2. КРОК 6-7. ПОШУК І УТОЧНЕННЯ КУТІВ ШАХІВНИЦІ
# ======================================================================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(img):
    """Повертає (кути без уточнення, кути з субпіксельним уточненням) або (None, None)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = min(1.0, DETECT_MAX_SIDE / max(gray.shape))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    ok, c = cv2.findChessboardCorners(small, PATTERN,
                                      cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None, None
    raw = (c / s).astype(np.float32)
    fine = cv2.cornerSubPix(gray, raw.copy(), (11, 11), (-1, -1), criteria)
    return raw, fine


raw_list, fine_list, ok_names, ok_imgs = [], [], [], []
for img, name in zip(imgs, names):
    raw, fine = find_corners(img)
    if raw is None:
        print("  кути НЕ знайдено:", name)
        continue
    raw_list.append(raw)
    fine_list.append(fine)
    ok_names.append(name)
    ok_imgs.append(img)
    vis = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.drawChessboardCorners(vis, PATTERN, fine * 0.25, True)
    cv2.imwrite(os.path.join(FIG, "corners", name), vis)
M = len(fine_list)
print(f"Кути знайдено на {M} з {len(imgs)} кадрів (картинки: figures/corners/)")
assert M >= 5, "Замало кадрів з виявленою шахівницею (потрібно щонайменше 5, краще 15-25)"

# ======================================================================
# 3. КРОК 8. ТРИВИМІРНІ КООРДИНАТИ КУТІВ ШАБЛОНУ
# ======================================================================
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM


# ======================================================================
# 4. КРОК 9-10. КАЛІБРУВАННЯ І ПОХИБКА РЕПРОЕКЦІЇ
# ======================================================================
def dists_for(K, dist, rvec, tvec, corners):
    """Відстані (пікселі) між виявленими кутами і тим, куди їх «проєктує» модель камери."""
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)


def rms(dists):
    """Середньоквадратична похибка по списку масивів відстаней (вона ж eRMS з лекції)."""
    return float(np.sqrt(np.mean(np.concatenate(dists) ** 2)))


def calibrate(corners_list):
    """Калібрує камеру за списком кутів. Повертає словник з K, дисторсією і похибками."""
    obj = [objp] * len(corners_list)
    rms_cv, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, corners_list, size, None, None)
    ds = [dists_for(K, dist, rv, tv, c) for rv, tv, c in zip(rvecs, tvecs, corners_list)]
    return dict(K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, dists=ds, rms=rms(ds), rms_cv=rms_cv)


print("\n=== Крок 7. Вплив субпіксельного уточнення кутів ===")
cal_raw, cal_fine = calibrate(raw_list), calibrate(fine_list)
print(f"без cornerSubPix: eRMS = {cal_raw['rms']:.3f} пкс | з cornerSubPix: eRMS = {cal_fine['rms']:.3f} пкс "
      f"(різниця у {cal_raw['rms'] / cal_fine['rms']:.1f} раза)")
print("  (кути 'без уточнення' знайдено на зменшеній копії, тому їх точність ~1/масштаб пікселя)")

print("\n=== Кроки 9-10. Калібрування на всіх кадрах ===")
cal = cal_fine
frame_err = np.array([d.mean() for d in cal["dists"]])
print(f"eRMS (власна) = {cal['rms']:.3f} пкс | eRMS, яку повернув OpenCV = {cal['rms_cv']:.3f} пкс")
fig = plt.figure(figsize=(10, 4))
plt.bar(range(M), frame_err)
plt.axhline(OUTLIER_FACTOR * np.median(frame_err), color="r", ls="--", label=f"поріг = {OUTLIER_FACTOR:g} × медіана")
plt.xticks(range(M), [n.replace(".jpg", "") for n in ok_names], rotation=90, fontsize=7)
plt.xlabel("кадр")
plt.ylabel("середня похибка репроєкції, пікселі")
plt.legend()
save_fig(fig, "per_frame_error.png")
save_csv("per_frame_error.csv", ["кадр", "середня похибка, пкс"],
         [[n, round(float(e), 3)] for n, e in zip(ok_names, frame_err)])

# ---- Крок 11. Вилучення викидів за критерієм, описаним ЗАЗДАЛЕГІДЬ ----
thr = OUTLIER_FACTOR * np.median(frame_err)
good = [i for i in range(M) if frame_err[i] <= thr]
bad = [ok_names[i] for i in range(M) if frame_err[i] > thr]
print(f"\n=== Крок 11. Викиди: кадр вилучається, якщо похибка > {OUTLIER_FACTOR:g} × медіана = {thr:.3f} пкс ===")
print(f"вилучено {len(bad)} кадр(ів): {bad if bad else '-'}; залишилось {len(good)}")
good_corners = [fine_list[i] for i in good]
cal2 = calibrate(good_corners)
print(f"eRMS: {cal['rms']:.3f} -> {cal2['rms']:.3f} пкс | fx: {cal['K'][0, 0]:.1f} -> {cal2['K'][0, 0]:.1f}")

# ---- Фінальні параметри ----
K, dist, final_rms = cal2["K"], cal2["dist"], cal2["rms"]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
print("\n=== Фінальна матриця камери K ===")
print(np.round(K, 2))
print("коефіцієнти дисторсії (k1, k2, p1, p2, k3):", np.round(dist.ravel(), 5))

# ---- Самоперевірка за орієнтирами з практикуму ----
print("\n=== Самоперевірка (орієнтири з практикуму; відхилення - привід пояснити, а не помилка) ===")
print(f"eRMS = {final_rms:.3f} пкс (орієнтир: до 0.5)")
print(f"fx/fy = {fx / fy:.4f} (орієнтир: 1.00 ± 0.01)")
print(f"головна точка зсунута від центра на {abs(cx - size[0] / 2) / size[0] * 100:.1f} % по x "
      f"і {abs(cy - size[1] / 2) / size[1] * 100:.1f} % по y (орієнтир: кілька %)")
try:  # оцінка fx за EXIF: fx ≈ f(екв. 35 мм) / 36 мм * ширина кадру в пікселях
    f35 = Image.open(paths[0]).getexif().get_ifd(0x8769).get(41989)
    fx_exif = float(f35) / 36.0 * max(size)
    print(f"fx за EXIF (екв. {f35} мм) ≈ {fx_exif:.0f} пкс, з калібрування {fx:.0f} пкс "
          f"(розбіжність {abs(fx - fx_exif) / fx_exif * 100:.1f} %)")
except Exception:
    print("EXIF з еквівалентною фокусною відстанню не знайдено - порівняння з fx пропущено")

# ======================================================================
# 5. КРОК 12. СКІЛЬКИ КАДРІВ ПОТРІБНО?
# ======================================================================
print("\n=== Крок 12. Залежність від кількості кадрів (випадкові підвибірки з очищених кадрів) ===")
pool = list(range(len(good_corners)))
rows, n_used = [], [n for n in N_LIST if n <= len(pool)]
stats = []
for n in n_used:
    tr, te, fxs = [], [], []
    for _ in range(REPEATS):
        idx = rng.choice(pool, n, replace=False)
        c = calibrate([good_corners[i] for i in idx])
        tr.append(c["rms"])
        fxs.append(c["K"][0, 0])
        rest = [i for i in pool if i not in idx]
        if rest:
            dd = []
            for i in rest:
                ok, rv, tv = cv2.solvePnP(objp, good_corners[i], c["K"], c["dist"])
                dd.append(dists_for(c["K"], c["dist"], rv, tv, good_corners[i]))
            te.append(rms(dd))
    stats.append((n, np.mean(tr), np.std(tr), np.mean(te) if te else np.nan, np.std(te) if te else np.nan,
                  np.mean(fxs), np.std(fxs)))
    print(f"N = {n:2d}: eRMS(на цих кадрах) {stats[-1][1]:.3f} ± {stats[-1][2]:.3f} | "
          f"eRMS(на інших кадрах) {stats[-1][3]:.3f} ± {stats[-1][4]:.3f} | fx {stats[-1][5]:.1f} ± {stats[-1][6]:.1f}")
S = np.array(stats)
save_csv("calib_vs_n.csv", ["N", "eRMS train", "std", "eRMS test", "std", "fx", "std fx"], np.round(S, 3).tolist())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(S[:, 0], S[:, 1], S[:, 2], marker="o", capsize=3, label="на кадрах калібрування")
ax[0].errorbar(S[:, 0], S[:, 3], S[:, 4], marker="s", capsize=3, label="на інших кадрах")
ax[0].set_xlabel("кількість кадрів N")
ax[0].set_ylabel("похибка репроєкції, пікселі")
ax[0].legend()
ax[1].errorbar(S[:, 0], S[:, 5], S[:, 6], marker="o", capsize=3)
ax[1].set_xlabel("кількість кадрів N")
ax[1].set_ylabel("оцінка fx, пікселі")
for a in ax:
    a.grid(alpha=.3)
save_fig(fig, "calib_vs_n.png")

# ======================================================================
# 6. КРОК 13-14. УСУНЕННЯ ДИСТОРСІЇ
# ======================================================================
line_path = os.path.join(CALIB, "..", "line_test.jpg")
line_img = cv2.imread(line_path) if os.path.exists(line_path) else None
if line_img is None:
    print("\nline_test.jpg немає - беру перший калібрувальний кадр (для кроку 14 потрібен кадр з прямою лінією біля краю)")
    line_img = ok_imgs[0]
assert (line_img.shape[1], line_img.shape[0]) == size, "line_test.jpg має бути такого ж розміру, як калібрувальні кадри"


def small_view(im, side=1400):
    s = side / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def with_line(im, p1, p2):
    out = im.copy()
    cv2.line(out, tuple(map(int, p1)), tuple(map(int, p2)), (0, 0, 255), max(3, size[0] // 600))
    return out


if LINE_PTS is None:
    v, s = small_view(line_img)
    fig = plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.grid(color="yellow", alpha=.5)
    plt.xlabel("x, піксель оригіналу")
    plt.ylabel("y, піксель оригіналу")
    save_fig(fig, "line_test_coords.png")
    print("\nLINE_PTS не задано: відкрийте figures/line_test_coords.png, зніміть координати двох кінців прямої "
          "(край одвірка, столу тощо) і впишіть у LINE_PTS на початку файлу. Лінію буде намальовано при наступному запуску.")

print("\n=== Кроки 13-14. Усунення дисторсії ===")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
panels = [("оригінал", line_img)]
for alpha in (0, 1):
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    und = cv2.undistort(line_img, K, dist, None, newK)
    black = np.mean(und.max(axis=2) == 0) * 100
    print(f"alpha={alpha}: чорні області займають {black:.1f} % кадру")
    panels.append((f"undistort, alpha={alpha}", und))
    if LINE_PTS is not None:
        pts = cv2.undistortPoints(np.array([LINE_PTS], np.float32), K, dist, P=newK).reshape(-1, 2)
        panels[-1] = (panels[-1][0], with_line(und, pts[0], pts[1]))
if LINE_PTS is not None:
    panels[0] = ("оригінал", with_line(line_img, *LINE_PTS))
for a, (title, im) in zip(ax, panels):
    a.imshow(cv2.cvtColor(small_view(im)[0], cv2.COLOR_BGR2RGB))
    a.set_title(title)
    a.axis("off")
save_fig(fig, "undistort.png")

# ======================================================================
# 7. КРОК 15. ЗБЕРЕЖЕННЯ ПАРАМЕТРІВ
# ======================================================================
yml = os.path.join(RES, "camera_params.yml")
fs = cv2.FileStorage(yml, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", K)
fs.write("dist_coeffs", dist)
fs.write("image_width", size[0])
fs.write("image_height", size[1])
fs.write("reprojection_error_px", final_rms)
fs.write("frames_used", len(good_corners))
fs.write("square_mm", float(SQUARE_MM))
fs.write("date", datetime.date.today().isoformat())
fs.release()
fs = cv2.FileStorage(yml, cv2.FILE_STORAGE_READ)
assert np.allclose(fs.getNode("camera_matrix").mat(), K)
fs.release()
print(f"\nПараметри збережено: {yml}")

save_csv("calibration_summary.csv", ["показник", "значення"], [
    ["розмір кадру", f"{size[0]}x{size[1]}"], ["кадрів знято", len(paths)], ["кути знайдено", M],
    ["вилучено викидів", len(bad)], ["кадрів використано", len(good_corners)],
    ["eRMS без cornerSubPix, пкс", round(cal_raw["rms"], 3)], ["eRMS з cornerSubPix, пкс", round(cal_fine["rms"], 3)],
    ["eRMS до вилучення викидів, пкс", round(cal["rms"], 3)], ["eRMS фінальна, пкс", round(final_rms, 3)],
    ["fx", round(fx, 2)], ["fy", round(fy, 2)], ["fx/fy", round(fx / fy, 4)], ["cx", round(cx, 2)], ["cy", round(cy, 2)],
    ["k1", round(float(dist.ravel()[0]), 5)], ["k2", round(float(dist.ravel()[1]), 5)],
    ["p1", round(float(dist.ravel()[2]), 5)], ["p2", round(float(dist.ravel()[3]), 5)],
    ["k3", round(float(dist.ravel()[4]), 5)], ["розмір клітини, мм", SQUARE_MM]])