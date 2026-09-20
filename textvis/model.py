"""Модель чтения строки интерфейса.

Архитектура CRNN — свёртки сжимают картинку по высоте, оставляя ширину,
затем двунаправленная GRU читает получившуюся последовательность колонок,
а CTC позволяет учиться без разметки «где какая буква».

Почему не полносвязная голова, как в Mine-AI: там вход разворачивался в
один вектор, и при 1280x720 это давало 51 млн параметров в одном слое,
уничтожая пространственную структуру. Для чтения нужна именно
последовательность, поэтому картинка остаётся лентой колонок.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .render_text import LINE_H
from .vocab import N_CHARS, N_LABELS


class TextReader(nn.Module):
    """Строка интерфейса -> последовательность символов и класс подписи."""

    def __init__(self, n_chars: int = N_CHARS, n_labels: int = N_LABELS,
                 hidden: int = 160):
        super().__init__()
        # Свёртки: высота 32 -> 1, ширина делится на 4.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.MaxPool2d(2, 2),                       # 16 x W/2
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.MaxPool2d(2, 2),                       # 8 x W/4
            nn.Conv2d(96, 144, 3, padding=1), nn.BatchNorm2d(144), nn.SiLU(),
            nn.MaxPool2d((2, 1), (2, 1)),             # 4 x W/4
            nn.Conv2d(144, 192, 3, padding=1), nn.BatchNorm2d(192), nn.SiLU(),
            nn.MaxPool2d((2, 1), (2, 1)),             # 2 x W/4
            nn.Conv2d(192, 192, (2, 1)), nn.SiLU(),   # 1 x W/4
        )
        self.rnn = nn.GRU(192, hidden, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.1)
        self.out_char = nn.Linear(hidden * 2, n_chars)
        # Вторая голова: узнать подпись целиком. Для известных пунктов меню
        # это надёжнее посимвольного чтения и даёт агенту готовый ответ.
        self.out_label = nn.Linear(hidden * 2, n_labels + 1)   # +1 = «иное»

    def forward(self, x):
        """x: (B, 3, 32, W) -> (log-probs символов, логиты подписи)."""
        f = self.conv(x)                       # (B, 192, 1, W/4)
        f = f.squeeze(2).permute(0, 2, 1)      # (B, W/4, 192)
        seq, _ = self.rnn(f)                   # (B, T, hidden*2)
        chars = self.out_char(seq).log_softmax(dim=2)
        label = self.out_label(seq.mean(dim=1))
        return chars, label

    @torch.no_grad()
    def read(self, x) -> list[str]:
        """Удобный разбор: картинки -> строки."""
        from .vocab import decode
        chars, _ = self(x)
        best = chars.argmax(dim=2).cpu().numpy()
        return [decode(row) for row in best]
