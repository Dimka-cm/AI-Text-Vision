"""Тесты чтения текста интерфейса."""
from __future__ import annotations

import numpy as np
import pytest

from textvis.render_text import (LINE_H, MAX_W, fonts_available, render_line,
                                 to_fixed)
from textvis.vocab import (ALPHABET, APPS, LABELS, N_CHARS, N_LABELS, decode,
                           decode_ids, encode)


def test_fonts_present():
    """Без настоящих шрифтов учить нечему."""
    f = fonts_available()
    assert len(f) >= 3, f"шрифтов мало: {f}"
    assert any("Roboto" in x for x in f), "нет Roboto — шрифта интерфейса UE5"


def test_alphabet_covers_both_languages():
    assert "A" in ALPHABET and "z" in ALPHABET
    assert "Д" in ALPHABET and "я" in ALPHABET
    assert "0" in ALPHABET and "." in ALPHABET
    assert N_CHARS == len(ALPHABET) + 1          # +1 на пустой символ CTC


def test_encode_decode_roundtrip():
    """Кодирование должно быть обратимым, включая двойные буквы."""
    for s in ["Save All", "Collision", "Настройки", "file01.uasset", "-12.5"]:
        assert decode_ids(encode(s)) == s


def test_ctc_decode_collapses_repeats():
    """CTC схлопывает повторы — это ожидаемое поведение, не баг."""
    assert decode(encode("Add")) == "Ad"


def test_all_six_apps_covered():
    """Агент работает в шести программах, словарь должен знать все."""
    for app in ("ue5", "ue4", "blender", "blockbench", "browser", "explorer"):
        assert app in APPS and len(APPS[app]) >= 15
    assert N_LABELS == len(LABELS) >= 200


def test_render_line_shape():
    s = render_line(np.random.default_rng(0))
    assert s.img.shape[0] == LINE_H
    assert s.img.shape[2] == 3
    assert s.img.dtype == np.uint8
    assert 9 <= s.font_px <= 15


def test_text_is_actually_drawn():
    """На картинке должен быть виден текст, а не пустой фон."""
    for seed in range(8):
        s = render_line(np.random.default_rng(seed), text="Save All")
        g = s.img.astype(np.float32).mean(axis=2)
        assert g.std() > 12, f"строка пустая, разброс {g.std():.1f}"


def test_to_fixed_pads_not_crops_short():
    s = render_line(np.random.default_rng(1), text="OK")
    out = to_fixed(s, MAX_W)
    assert out.shape == (LINE_H, MAX_W, 3)


def test_both_dark_and_light_themes():
    """Проводник и браузер светлые, редакторы тёмные — нужны оба вида."""
    rng = np.random.default_rng(5)
    dark = np.mean([render_line(rng, app="ue5").img.mean() for _ in range(8)])
    light = np.mean([render_line(rng, app="explorer").img.mean()
                     for _ in range(8)])
    assert dark < 110, f"тёмная тема слишком светлая: {dark:.0f}"
    assert light > 120, f"светлая тема слишком тёмная: {light:.0f}"


def test_cyrillic_renders():
    s = render_line(np.random.default_rng(2), text="Рабочий стол",
                    app="explorer")
    g = s.img.astype(np.float32).mean(axis=2)
    assert g.std() > 12, "кириллица не отрисовалась"


def test_model_shapes():
    torch = pytest.importorskip("torch")
    from textvis.model import TextReader
    m = TextReader()
    ch, lb = m(torch.zeros(2, 3, LINE_H, MAX_W))
    assert ch.shape[0] == 2 and ch.shape[2] == N_CHARS
    assert ch.shape[1] == MAX_W // 4, "шагов по ширине должно быть W/4"
    assert lb.shape == (2, N_LABELS + 1)


def test_model_is_small_enough():
    """Читалка должна быть лёгкой: она вызывается на каждый элемент экрана."""
    torch = pytest.importorskip("torch")
    from textvis.model import TextReader
    n = sum(p.numel() for p in TextReader().parameters())
    assert n < 4_000_000, f"модель разрослась: {n:,}"


def test_train_and_test_words_never_overlap():
    """Словари обучения и проверки не должны пересекаться.

    Прошлый прогон провалился именно здесь: 93% проверочных строк дословно
    встречались в обучении, потеря упала до 0.0067, и «98.5% точности»
    означали лишь заученный список из 254 слов.
    """
    from textvis.vocab import HELD_OUT, TRAIN_WORDS
    assert not (set(TRAIN_WORDS) & set(HELD_OUT))
    assert len(HELD_OUT) >= 40, "отложено слишком мало слов"
    assert len(TRAIN_WORDS) >= 150


def test_generated_names_differ_between_splits():
    """Имена ассетов в проверке должны строиться по другим шаблонам."""
    from textvis.render_text import _rand_name
    rng = np.random.default_rng(0)
    tr = {_rand_name(rng, "train") for _ in range(400)}
    te = {_rand_name(rng, "test") for _ in range(400)}
    common = tr & te
    assert not common, f"совпадающие имена: {list(common)[:5]}"


def test_test_split_produces_unseen_text():
    """Выборка проверки не должна дословно повторять обучающую."""
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    train = {render_line(rng_a, split="train").text for _ in range(600)}
    test = [render_line(rng_b, split="test").text for _ in range(600)]
    leak = sum(1 for t in test if t in train) / len(test)
    assert leak < 0.15, f"утечка {leak * 100:.0f}% строк проверки"


def test_random_text_is_a_big_share():
    """Случайных строк должно быть много: на них нечего заучивать."""
    from textvis.vocab import APPS
    rng = np.random.default_rng(3)
    words = {w for lst in APPS.values() for w in lst}
    texts = [render_line(rng, split="train").text for _ in range(500)]
    rand = sum(1 for t in texts if t not in words) / len(texts)
    assert rand > 0.45, f"случайного текста лишь {rand * 100:.0f}%"
