"""Генератор строк интерфейса для обучения чтению.

Каждый кадр — это одна надпись так, как она выглядит на экране: тот же
шрифт, тот же кегль, тот же фон панели, то же сглаживание. Модель учится
читать её посимвольно.

Почему не берём готовый датасет OCR: там фотографии вывесок и сканы
документов. Интерфейс выглядит иначе — мелкий экранный шрифт с
субпиксельным сглаживанием на однотонном фоне, и именно к нему нужно
приучить сеть.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .vocab import ALPHABET, APPS, APPS_TEST, APPS_TRAIN

FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# Высота строки, в которую приводится любой вырез перед чтением.
LINE_H = 32
MAX_W = 256

# Темы шести программ: фон панели, цвет текста, цвет выделения.
# Значения взяты из настоящих интерфейсов: тёмная тема Epic/Blender,
# светлая тема проводника Windows 11 и браузеров.
THEMES = {
    "ue5":        ((36, 36, 36),   (192, 192, 192), (0, 112, 224)),
    "ue5_header": ((47, 47, 47),   (255, 255, 255), (0, 112, 224)),
    "ue4":        ((62, 62, 62),   (200, 200, 200), (16, 94, 168)),
    "blender":    ((48, 48, 48),   (218, 218, 218), (71, 114, 179)),
    "blender_hi": ((58, 58, 58),   (255, 255, 255), (71, 114, 179)),
    "blockbench": ((41, 44, 51),   (211, 214, 219), (51, 122, 183)),
    "browser":    ((255, 255, 255), (32, 33, 36),   (26, 115, 232)),
    "browser_dk": ((32, 33, 36),   (232, 234, 237), (138, 180, 248)),
    "explorer":   ((243, 243, 243), (26, 26, 26),   (0, 103, 192)),
    "explorer_dk": ((32, 32, 32),  (255, 255, 255), (76, 194, 255)),
}

APP_THEME = {
    "ue5": ["ue5", "ue5_header"], "ue4": ["ue4"],
    "blender": ["blender", "blender_hi"], "blockbench": ["blockbench"],
    "browser": ["browser", "browser_dk"],
    "explorer": ["explorer", "explorer_dk"],
}


@lru_cache(maxsize=64)
def _font(name: str, size: int):
    return ImageFont.truetype(str(FONTS / name), size)


def fonts_available() -> list[str]:
    return sorted(p.name for p in FONTS.glob("*.ttf"))


@dataclass
class TextSample:
    """Вырез со строкой и то, что в ней написано."""
    img: np.ndarray          # (LINE_H, W, 3) uint8
    text: str
    app: str
    font_px: int


def _rand_name(rng, split: str) -> str:
    """Случайное имя ассета или файла.

    Шаблоны у обучения и проверки РАЗНЫЕ. Иначе модель заучит десяток
    сочетаний вроде SM_001 и будет узнавать их целиком, а на живом
    SM_Rock_Cliff_02 растеряется.
    """
    if split == "train":
        pre = ["SM", "BP", "M", "T", "MI", "SK", "AM", "NS", "WBP"]
        mid = ["Wall", "Floor", "Door", "Lamp", "Tree", "Rock", "Metal",
               "Wood", "Glass", "Water", "Sand", "Brick", "Panel", "Pipe"]
        suf = ["01", "02", "A", "B", "Base", "Inst", "LOD0", "New", "Old"]
    else:
        pre = ["SMesh", "Blue", "Mat", "Tex", "Char", "Env", "FX", "UI"]
        mid = ["Cliff", "Canyon", "Statue", "Bridge", "Fence", "Barrel",
               "Crate", "Torch", "Banner", "Carpet", "Mirror", "Anvil"]
        suf = ["03", "07", "C", "D", "Final", "Draft", "LOD2", "Alt"]

    parts = [str(rng.choice(pre))]
    for _ in range(int(rng.integers(1, 3))):
        parts.append(str(rng.choice(mid)))
    if rng.random() < 0.7:
        parts.append(str(rng.choice(suf)))
    return "_".join(parts)


def _rand_gibberish(rng) -> str:
    """Просто случайные буквы.

    Самая важная часть выборки: на бессмысленных строках заучивать нечего,
    остаётся только честно читать символ за символом. Без них сеть быстро
    скатывается в узнавание слов целиком.
    """
    n = int(rng.integers(3, 15))
    pool = ALPHABET.replace(" ", "")
    if rng.random() < 0.5:                       # только латиница
        pool = "".join(c for c in pool if c.isascii())
    out = []
    for i in range(n):
        out.append(str(rng.choice(list(pool))))
        if rng.random() < 0.08 and 0 < i < n - 1:
            out.append(" ")
    return "".join(out)


def _pick_text(rng, app: str | None = None,
               split: str = "train") -> tuple[str, str]:
    """Берёт надпись: из словаря нужной половины либо генерирует."""
    table = APPS_TRAIN if split == "train" else APPS_TEST
    app = app or str(rng.choice([k for k in table if table[k]]))
    words = table.get(app) or APPS[app]
    r = rng.random()
    if r < 0.42:
        return str(words[int(rng.integers(0, len(words)))]), app
    if r < 0.68:
        return _rand_name(rng, split), app
    if r < 0.88:
        return _rand_gibberish(rng), app
    if r < 0.96:
        digits = f"{rng.normal(0, 500):.{int(rng.integers(0, 4))}f}"
        return digits, app
    ext = ([".uasset", ".blend", ".png", ".fbx"] if split == "train"
           else [".bbmodel", ".txt", ".jpg", ".obj"])
    return (f"{_rand_name(rng, split)}{ext[int(rng.integers(0, len(ext)))]}",
            app)


def render_line(rng=None, app: str | None = None,
                text: str | None = None,
                split: str = "train") -> TextSample:
    """Рисует одну строку интерфейса со случайным оформлением."""
    if rng is None:
        rng = np.random.default_rng()
    if text is None:
        text, app = _pick_text(rng, app, split)
    else:
        app = app or "ue5"

    theme_name = APP_THEME.get(app, ["ue5"])[
        int(rng.integers(0, len(APP_THEME.get(app, ["ue5"]))))]
    bg, fg, sel = THEMES[theme_name]

    # Кегль в пикселях. На входе 1280x720 надписи занимают 9-15 px.
    size = int(rng.integers(9, 16))
    names = fonts_available()
    fname = names[int(rng.integers(0, len(names)))]
    font = _font(fname, size)

    pad_x = int(rng.integers(3, 9))
    bbox = font.getbbox(text)
    tw = max(6, bbox[2] - bbox[0])
    w = min(MAX_W, tw + pad_x * 2)

    # Рисуем в двойном масштабе и уменьшаем — так получается сглаживание,
    # похожее на экранное, а не рваные края.
    S = 2
    im = Image.new("RGB", (w * S, LINE_H * S), bg)
    d = ImageDraw.Draw(im)
    if rng.random() < 0.18:                       # строка выделена
        d.rectangle([0, 0, w * S, LINE_H * S], fill=sel)
        fg = (255, 255, 255)
    y = (LINE_H * S - (bbox[3] - bbox[1]) * S) // 2 - bbox[1] * S
    big = _font(fname, size * S)
    d.text((pad_x * S, y), text, font=big, fill=fg)
    im = im.resize((w, LINE_H), Image.LANCZOS)

    a = np.asarray(im).astype(np.float32)
    a += rng.normal(0, 1.6, a.shape)              # шум матрицы и сжатия
    np.clip(a, 0, 255, out=a)
    return TextSample(img=a.astype(np.uint8), text=text, app=app,
                      font_px=size)


def to_fixed(sample: TextSample, width: int = MAX_W) -> np.ndarray:
    """Приводит вырез к постоянной ширине: так их можно собрать в батч."""
    img = sample.img
    h, w = img.shape[:2]
    out = np.zeros((h, width, 3), np.uint8)
    out[:] = img[0, 0]                            # дополняем цветом фона
    out[:, :min(w, width)] = img[:, :min(w, width)]
    return out
