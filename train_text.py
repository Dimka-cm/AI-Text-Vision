#!/usr/bin/env python3
"""
Обучение чтению текста интерфейса.

    python3 train_text.py --samples 40000 --epochs 20

Быстрая проверка:
    python3 train_text.py --samples 2000 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from textvis.model import TextReader
from textvis.render_text import LINE_H, MAX_W, render_line, to_fixed
from textvis.vocab import LABEL_ID, N_LABELS, decode, encode


def collect(n, rng, width=MAX_W, verbose=True, split="train"):
    """Собирает n строк с подписями."""
    X = np.zeros((n, LINE_H, width, 3), np.uint8)
    texts, apps, labels = [], [], np.zeros(n, np.int64)
    t0 = time.time()
    for i in range(n):
        s = render_line(rng, split=split)
        X[i] = to_fixed(s, width)
        texts.append(s.text)
        apps.append(s.app)
        labels[i] = LABEL_ID.get(s.text, N_LABELS)     # N_LABELS = «иное»
        if verbose and (i + 1) % max(1, n // 8) == 0:
            el = time.time() - t0
            print(f"  собрано {i + 1}/{n} ({el:.0f} с, "
                  f"осталось ~{el / (i + 1) * (n - i - 1):.0f} с)", flush=True)
    return X, texts, apps, labels


def _batch_targets(texts):
    """Готовит цели для CTC: склеенные номера символов и их длины."""
    seqs = [encode(t) for t in texts]
    flat = torch.tensor([c for s in seqs for c in s], dtype=torch.long)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return flat, lens


def evaluate(model, X, texts, labels, batch=64):
    """Доля точно прочитанных строк, посимвольная точность и узнавание."""
    model.eval()
    exact = chars_ok = chars_tot = lab_ok = 0
    per_app = {}
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(
                X[i:i + batch].transpose(0, 3, 1, 2)).float().div_(255.)
            ch, lg = model(xb)
            pred = ch.argmax(dim=2).cpu().numpy()
            lab = lg.argmax(dim=1).cpu().numpy()
            for k, row in enumerate(pred):
                got, want = decode(row), texts[i + k]
                exact += (got == want)
                m = min(len(got), len(want))
                chars_ok += sum(a == b for a, b in zip(got[:m], want[:m]))
                chars_tot += max(len(got), len(want))
            lab_ok += (lab == labels[i:i + batch]).sum()
    model.train()
    return {"exact": exact / len(X), "chars": chars_ok / max(1, chars_tot),
            "label": lab_ok / len(X)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение чтению интерфейса")
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--test-samples", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--out", default="text_reader.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cores:
        torch.set_num_threads(args.cores)
    print(f"[cpu] потоков: {torch.get_num_threads()}")

    mb = (args.samples + args.test_samples) * LINE_H * MAX_W * 3 / 1e6
    print(f"Сбор: {args.samples} строк + {args.test_samples} проверочных")
    print(f"ОЗУ под кадры: ~{mb:.0f} МБ (строка весит "
          f"{LINE_H * MAX_W * 3 / 1024:.0f} КБ)")

    rng = np.random.default_rng(args.seed)
    X, texts, apps, labels = collect(args.samples, rng, split="train")
    # Две проверочные выборки. «Знакомая» — из тех же слов, что в обучении:
    # она показывает, запомнила ли модель. «Новая» — из отложенных 51 слова
    # и других шаблонов имён, которых сеть не видела ни разу: только она
    # говорит, умеет ли модель ЧИТАТЬ. В прошлом прогоне была лишь первая,
    # потому потеря и упала до 0.0067 при неизвестном реальном качестве.
    Xte, tte, ate, lte = collect(args.test_samples,
                                 np.random.default_rng(args.seed + 999),
                                 split="train", verbose=False)
    Xnew, tnew, anew, lnew = collect(args.test_samples,
                                     np.random.default_rng(args.seed + 555),
                                     split="test", verbose=False)

    model = TextReader()
    print(f"параметров: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    lab_t = torch.from_numpy(labels)

    curve = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(
                X[idx].transpose(0, 3, 1, 2)).float().div_(255.)
            ch, lg = model(xb)
            flat, lens = _batch_targets([texts[j] for j in idx])
            inp_len = torch.full((len(idx),), ch.shape[1], dtype=torch.long)
            loss = ctc(ch.permute(1, 0, 2), flat, inp_len, lens)
            loss = loss + 0.3 * F.cross_entropy(lg, lab_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        el = time.time() - t0
        line = (f"  эпоха {ep}/{args.epochs}  потеря {tot / len(X):.4f}  "
                f"({el:.0f} с, осталось ~{el / ep * (args.epochs - ep):.0f} с)")
        if args.eval_every and (ep % args.eval_every == 0
                                or ep == args.epochs):
            r = evaluate(model, Xte, tte, lte, args.batch)
            rn = evaluate(model, Xnew, tnew, lnew, args.batch)
            line += (f"  | знакомые {r['exact'] * 100:.1f}%  "
                     f"НОВЫЕ {rn['exact'] * 100:.1f}%  "
                     f"символы {rn['chars'] * 100:.1f}%")
            curve.append({"epoch": ep, "loss": tot / len(X),
                          "seen": r, "new": rn})
        print(line, flush=True)
        if args.eval_every and (ep % args.eval_every == 0
                                or ep == args.epochs):
            # Сохраняем после каждой проверки. Прошлый прогон писал веса
            # только в самом конце: когда его остановили на 9-й эпохе из 30,
            # полтора часа счёта пропали целиком, а в артефакте остался
            # один лог. Больше так не теряем.
            torch.save({"state": model.state_dict(), "epoch": ep,
                        "curve": curve}, args.out)

    rep = evaluate(model, Xte, tte, lte, args.batch)
    rep_new = evaluate(model, Xnew, tnew, lnew, args.batch)
    by_app = {}
    for a in sorted(set(anew)):
        sel = [i for i, x in enumerate(anew) if x == a]
        by_app[a] = evaluate(model, Xnew[sel], [tnew[i] for i in sel],
                             lnew[sel], args.batch)

    print("\n" + "=" * 64)
    print("ЧТО МОДЕЛЬ НАУЧИЛАСЬ ЧИТАТЬ")
    print("=" * 64)
    print(f"{'выборка':34s} {'строки':>10s} {'символы':>10s}")
    print("-" * 64)
    print(f"{'знакомые слова (была в обучении)':34s} "
          f"{rep['exact'] * 100:9.1f}% {rep['chars'] * 100:9.1f}%")
    print(f"{'НОВЫЕ слова (не видела ни разу)':34s} "
          f"{rep_new['exact'] * 100:9.1f}% {rep_new['chars'] * 100:9.1f}%")
    print("-" * 64)
    gap = (rep['exact'] - rep_new['exact']) * 100
    print(f"разрыв: {gap:.1f} процентных пункта", end="  ")
    print("(большой разрыв = зубрёжка, а не чтение)")
    print("\nНОВЫЕ СЛОВА ПО ПРОГРАММАМ")
    print("-" * 64)
    for a, r in by_app.items():
        print(f"  {a:14s} строки {r['exact'] * 100:6.1f}%   "
              f"символы {r['chars'] * 100:6.1f}%")
    print("=" * 64)

    print("\nПРИМЕРЫ ЧТЕНИЯ (строки, которых модель не видела):")
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(
            Xnew[:14].transpose(0, 3, 1, 2)).float().div_(255.)
        got = model.read(xb)
    for g, w in zip(got, tnew[:14]):
        mark = "верно " if g == w else "ОШИБКА"
        print(f"  {mark}  ожидалось {w!r:26s} прочитано {g!r}")

    torch.save({"state": model.state_dict(), "report": rep,
                "report_new": rep_new, "by_app": by_app,
                "curve": curve}, args.out)
    json.dump({"report_seen": rep, "report_new": rep_new,
               "by_app": by_app, "curve": curve},
              open(args.out.replace(".pt", ".json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\nСохранено: {args.out}")
    print(f"Заняло: {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
