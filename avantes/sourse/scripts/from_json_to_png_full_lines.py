import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import glob

# ---------------------------------------------------------------------------
# Импорт ваших функций из source/scripts/after_shot_analysys.py
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, "..", "scripts"))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from after_shot_analysys import data_negatives_to_zeros, waves_search
import avaread

times = 15

# ---------------------------------------------------------------------------
# Списки линий и групп (теоретические длины волн)
# ---------------------------------------------------------------------------
GROUP_CIII_OII = [464.728, 464.916, 465.025, 465.147]
GROUP_OII      = [434.74, 435.139]
GROUP_CII      = [657.8, 658.28]

SINGLE_CRI   = [425.4331, 427.4806, 428.9733]
SINGLE_NII   = [568.125]
SINGLE_DEUT  = [434.0625, 485.9375, 656.25]

OUT_DIR = r"C:\Users\elena\Documents\GitHub\spec\avantes\out_data"
TOL = 0.5


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def shot_name(data_directory):
    base = os.path.basename(data_directory)
    return base.rsplit(".", 1)[0]


def get_exact_waves(data_directory, theoretical_waves):
    pre_data = avaread.read_file(data_directory)
    waves = pre_data.wavelength
    out = []
    for w in theoretical_waves:
        if waves.min() - 2 <= w <= waves.max() + 2:
            out.append(float(waves_search(waves, [w])[0]))
    return out


def load_noise(channel):
    if channel == "blue":
        path = os.path.join(OUT_DIR, "errors_blue_Noise blue absolute.json")
    else:
        path = os.path.join(OUT_DIR, "errors_red_Noise red absolute.json")
    if not os.path.exists(path):
        print(f"[noise] нет файла {path}")
        return None
    with open(path, "r") as f:
        return np.asarray(json.load(f), dtype=float)


def list_shot_files(data_directory, kind):
    fname = shot_name(data_directory)
    prefix = "params_full_frames" if kind == "params" else "area_full_frames"
    return glob.glob(os.path.join(OUT_DIR, f"{prefix}_*_{fname}.json"))


def find_file_by_key(data_directory, kind, target_wave, tol=TOL):
    for path in list_shot_files(data_directory, kind):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not data:
                continue
            keys = list(data.keys())
            vals = np.array([float(k) for k in keys])
            diffs = np.abs(vals - target_wave)
            idx = int(np.argmin(diffs))
            if diffs[idx] <= tol:
                return data, keys[idx], path
        except Exception:
            continue
    return None, None, None


def extract_area(data, key):
    entry = data[key]
    area = np.array(entry["area_with_errors"]["area"], dtype=float)
    stderr = np.array(entry["area_with_errors"]["stderr"], dtype=float)
    return area, stderr


def combine_errors(area, stderr, noise_mean):
    area = np.asarray(area, dtype=float)
    stderr = np.asarray(stderr, dtype=float)
    fallback = np.full_like(area, noise_mean if noise_mean else 0.0, dtype=float)
    return np.where(np.isfinite(stderr) & (stderr > 0), stderr, fallback)


def sanitize_err(y, err, ratio=0.5):
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.all(~np.isfinite(err)):
        return np.zeros_like(y)
    max_y = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 0.0
    limit = max_y * ratio
    return np.where((err > limit) | (~np.isfinite(err)) | (err <= 0), 0.0, err)


def plot_series(ax, y, err, label, color):
    t = np.arange(len(y))
    ax.plot(t, y, color=color, linewidth=1.4, label=label)
    e = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    ax.fill_between(t, y + e, y - e, color=color, alpha=0.25)
    for i in t:
        if e[i] <= 0:
            ax.scatter(i, y[i], color="red", s=14, zorder=5)
        else:
            ax.scatter(i, y[i], color=color, s=14, zorder=5)


# ---------------------------------------------------------------------------
# Панель-обёртка: одиночная линия из одного канала
# ---------------------------------------------------------------------------
def draw_panel(ax, data_directory, kind, noise, names, colors, theor_waves, title):
    if noise is not None:
        noise_mean = float(np.nanmean(noise))
    else:
        noise_mean = None

    exact_waves = get_exact_waves(data_directory, theor_waves)
    if not exact_waves:
        ax.set_title(title + " (нет данных)", fontsize=9)
        ax.set_xlabel("Frame, n")
        ax.set_ylabel("Area, a. u.")
        ax.grid(True, alpha=0.4)
        return

    for t_wave, e_wave, name, color in zip(theor_waves, exact_waves, names, colors):
        data, key, path = find_file_by_key(data_directory, kind, e_wave, tol=TOL)
        if data is None:
            data, key, path = find_file_by_key(data_directory, kind, t_wave, tol=TOL)
        if data is None:
            print(f"  [skip] {name} {t_wave:.3f} — нет файла")
            continue
        area, stderr = extract_area(data, key)
        area = data_negatives_to_zeros(area)
        err = sanitize_err(area, combine_errors(area, stderr, noise_mean))
        plot_series(ax, area, err, f"{name} {t_wave:.2f} nm", color)
        print(f"  [ok]   {name} {t_wave:.3f} <- {os.path.basename(path)} (key {key})")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frame, n")
    ax.set_ylabel("Area, a. u.")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Специальная панель: C III + O II как СУММА четырёх линий
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Панель: 4 линии C III/O II ОТДЕЛЬНО на одном графике
# ---------------------------------------------------------------------------
def draw_cIII_oII_lines(ax, data_directory, noise, title):
    if noise is not None:
        noise_mean = float(np.nanmean(noise))
    else:
        noise_mean = None

    exact_waves = get_exact_waves(data_directory, GROUP_CIII_OII)
    if not exact_waves:
        ax.set_title(title + " (нет данных)", fontsize=9)
        ax.set_xlabel("Frame, n")
        ax.set_ylabel("Area, a. u.")
        ax.grid(True, alpha=0.4)
        return

    # порядок и подписи как в gauss_sum_of_CIII_OII
    # 464.728 C III, 464.916 O II, 465.025 C III, 465.147 C III
    names  = ["C III", "O II",  "C III", "C III"]
    colors = ["#CC0000", "#1F8CFF", "#2E8B57", "#7C1FC0"]

    drew_any = False
    for e_wave, t_wave, name, color in zip(exact_waves, GROUP_CIII_OII,
                                           names, colors):
        data, key, path = find_file_by_key(data_directory, "params", e_wave, tol=TOL)
        if data is None:
            data, key, path = find_file_by_key(data_directory, "params", t_wave, tol=TOL)
        if data is None:
            print(f"  [skip] {name} {t_wave:.3f} — нет файла")
            continue

        area, stderr = extract_area(data, key)
        area = data_negatives_to_zeros(area)
        err = sanitize_err(area, combine_errors(area, stderr, noise_mean))

        plot_series(ax, area, err, f"{name} {t_wave:.3f} nm", color)
        print(f"  [ok]   {name} {t_wave:.3f} <- {os.path.basename(path)} (key {key})")
        drew_any = True

    if not drew_any:
        ax.set_title(title + " (нет данных)", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frame, n")
    ax.set_ylabel("Area, a. u.")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7)

# ---------------------------------------------------------------------------
# Специальная панель: ВСЕ D-линии на одном графике.
# D gamma и D beta берём из blue_dir, D alpha — из red_dir.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Панель D-линий: D alpha на левой оси,
# D gamma / D beta — на правой оси (относительно D alpha)
# ---------------------------------------------------------------------------
def draw_all_D_lines(ax, blue_dir, red_dir, noise_blue, noise_red, title):
    noise_blue_mean = float(np.nanmean(noise_blue)) if noise_blue is not None else None
    noise_red_mean  = float(np.nanmean(noise_red))  if noise_red  is not None else None

    ax2 = ax.twinx()   # вторая ось справа

    # D alpha — левая ось (красный канал)
    # D gamma, D beta — правая ось (синий канал)
    specs_left = [
        (656.25, "D alpha", "#E6B800", red_dir,  noise_red_mean,  ax),
    ]
    specs_right = [
        (434.0625, "D gamma", "#CC0000", blue_dir, noise_blue_mean, ax2),
        (485.9375, "D beta",  "#FF6600", blue_dir, noise_blue_mean, ax2),
    ]

    drew_left = False
    for t_wave, name, color, d_dir, noise_mean, target_ax in specs_left:
        exact = get_exact_waves(d_dir, [t_wave])
        if not exact:
            print(f"  [skip] {name} {t_wave:.3f} — вне диапазона {os.path.basename(d_dir)}")
            continue
        e_wave = exact[0]
        data, key, path = find_file_by_key(d_dir, "area", e_wave, tol=TOL)
        if data is None:
            data, key, path = find_file_by_key(d_dir, "area", t_wave, tol=TOL)
        if data is None:
            print(f"  [skip] {name} {t_wave:.3f} — нет файла в {os.path.basename(d_dir)}")
            continue
        area, stderr = extract_area(data, key)
        area = data_negatives_to_zeros(area)
        err = sanitize_err(area, combine_errors(area, stderr, noise_mean))
        plot_series(target_ax, area, err, f"{name} {t_wave:.2f} nm", color)
        print(f"  [ok]   {name} {t_wave:.3f} <- {os.path.basename(path)} (key {key})")
        drew_left = True

    drew_right = False
    for t_wave, name, color, d_dir, noise_mean, target_ax in specs_right:
        exact = get_exact_waves(d_dir, [t_wave])
        if not exact:
            print(f"  [skip] {name} {t_wave:.3f} — вне диапазона {os.path.basename(d_dir)}")
            continue
        e_wave = exact[0]
        data, key, path = find_file_by_key(d_dir, "area", e_wave, tol=TOL)
        if data is None:
            data, key, path = find_file_by_key(d_dir, "area", t_wave, tol=TOL)
        if data is None:
            print(f"  [skip] {name} {t_wave:.3f} — нет файла в {os.path.basename(d_dir)}")
            continue
        area, stderr = extract_area(data, key)
        area = data_negatives_to_zeros(area)
        err = sanitize_err(area, combine_errors(area, stderr, noise_mean))
        plot_series(target_ax, area, err, f"{name} {t_wave:.2f} nm", color)
        print(f"  [ok]   {name} {t_wave:.3f} <- {os.path.basename(path)} (key {key})")
        drew_right = True

    if not drew_left and not drew_right:
        ax.set_title(title + " (нет данных)", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    # левая ось — D alpha
    ax.set_xlabel("Frame, n")
    ax.set_ylabel("D alpha: Area, a. u.", color="#B8860B")
    ax.tick_params(axis="y", labelcolor="#B8860B")
    ax.grid(True, alpha=0.4)

    # правая ось — D gamma / D beta
    ax2.set_ylabel("D gamma / D beta: Area, a. u.", color="#CC0000")
    ax2.tick_params(axis="y", labelcolor="#CC0000")

    # объединённая легенда с обеих осей
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")


# ---------------------------------------------------------------------------
# Основная визуализация
# ---------------------------------------------------------------------------
def visualize_two_channels(data_directory_1, data_directory_2):
    # p00 33 = КРАСНЫЙ канал, p00 45 = СИНИЙ канал
    red_dir  = data_directory_1
    blue_dir = data_directory_2

    red_shot  = shot_name(red_dir)
    blue_shot = shot_name(blue_dir)

    noise_blue = load_noise("blue")
    noise_red  = load_noise("red")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    # ================== ВЕРХНИЙ РЯД — СИНИЙ (p00 45) ==================
    print(f"\n=== BLUE: {blue_shot} ===")

    # 1) C III + O II (СУММА)
    print(f"\n-- C III + O II (сумма) --")
    draw_cIII_oII_lines(axes[0], blue_dir, noise_blue,
                        f"Blue {blue_shot}: C III / O II (4 линии)")

    # 2) O II
    print(f"\n-- O II --")
    draw_panel(axes[1], blue_dir, "params", noise_blue,
               ["O II", "O II"],
               ["#0033CC", "#1F8CFF"],
               GROUP_OII, f"Blue {blue_shot}: O II")

    # 3) Cr I
    print(f"\n-- Cr I --")
    draw_panel(axes[2], blue_dir, "area", noise_blue,
               ["Cr I", "Cr I", "Cr I"],
               ["#CC0000", "#E6B800", "#2E8B57"],
               SINGLE_CRI, f"Blue {blue_shot}: Cr I")

    # ================== НИЖНИЙ РЯД — КРАСНЫЙ (p00 33) ==================
    print(f"\n=== RED: {red_shot} ===")

    # 4) Все D-линии на ОДНОМ графике (D gamma/D beta — из синего, D alpha — из красного)
    print(f"\n-- D gamma / D beta / D alpha (объединено) --")
    draw_all_D_lines(axes[3], blue_dir, red_dir, noise_blue, noise_red,
                     f"D-линии: {blue_shot} + {red_shot}")

    # 5) N II
    print(f"\n-- N II --")
    draw_panel(axes[4], red_dir, "area", noise_red,
               ["N II"],
               ["#7C1FC0"],
               SINGLE_NII, f"Red {red_shot}: N II")

    # 6) C II
    print(f"\n-- C II --")
    draw_panel(axes[5], red_dir, "params", noise_red,
               ["C II", "C II"],
               ["#00A3CC", "#3C9E3C"],
               GROUP_CII, f"Red {red_shot}: C II")

    fig.suptitle(f"After-shot analysis: blue {blue_shot} vs red {red_shot}",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, f"after_shot_blue{blue_shot}_red{red_shot}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[saved] {out_path}")
    plt.show()


def main():
    data_directory_1 = (r'C:\Users\elena\PycharmProjects\PythonProject\.venv'
                        r'\FTI_work\avantes\data\261020\p00 34.STR8')
    data_directory_2 = (r'C:\Users\elena\PycharmProjects\PythonProject\.venv'
                        r'\FTI_work\avantes\data\261020\p00 46.STR8')
    visualize_two_channels(data_directory_1, data_directory_2)


if __name__ == "__main__":
    main()