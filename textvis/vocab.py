"""Словарь надписей интерфейса и алфавит для чтения.

Агент работает в шести программах, и почти весь их текст — это конечный
набор устойчивых подписей: пункты меню, кнопки, названия панелей, имена
свойств. Учить модель читать произвольный текст дороже и незачем: важно
надёжно узнавать то, по чему кликают.

Алфавит нужен для посимвольного чтения (CTC): им читаются имена файлов,
числа в полях и всё, чего нет в словаре.
"""
from __future__ import annotations

# Алфавит: латиница, кириллица, цифры, знаки, встречающиеся в интерфейсе.
ALPHABET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
)
BLANK = 0                                  # пустой символ для CTC
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ALPHABET)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
N_CHARS = len(ALPHABET) + 1


def encode(text: str) -> list[int]:
    """Строка -> номера символов. Незнакомые символы пропускаются."""
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


def decode(ids) -> str:
    """Номера -> строка по правилам CTC: схлопываем повторы, убираем пустые.

    Внимание: это разбор ВЫХОДА сети, где двойные буквы разделены пустым
    символом. Для обратной проверки encode/decode пользуйтесь decode_ids.
    """
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ID_TO_CHAR.get(i, ""))
        prev = i
    return "".join(out)


def decode_ids(ids) -> str:
    """Номера -> строка без схлопывания: точная обратная операция к encode."""
    return "".join(ID_TO_CHAR.get(int(i), "") for i in ids if int(i) != BLANK)


# -- надписи по программам ------------------------------------------------
# Взяты из реальных интерфейсов, а не придуманы: это то, по чему агент
# будет кликать.

UE5 = [
    "File", "Edit", "Window", "Tools", "Build", "Select", "Actor", "Help",
    "Save All", "Save Current Level", "Open Level", "New Level", "Import",
    "Export All", "Play", "Stop", "Pause", "Eject", "Simulate", "Compile",
    "Content Drawer", "Output Log", "World Settings", "Project Settings",
    "Details", "Outliner", "Content Browser", "Viewport", "Place Actors",
    "Blueprints", "Cinematics", "Add", "Settings", "Platforms",
    "Transform", "Location", "Rotation", "Scale", "Mobility",
    "Static", "Stationary", "Movable", "Materials", "Element 0",
    "Rendering", "Physics", "Collision", "Lighting", "Navigation",
    "Static Mesh", "Skeletal Mesh", "Material", "Texture", "Blueprint Class",
    "Level Sequence", "Niagara System", "Actor Label", "Folder Path",
    "Visible", "Hidden In Game", "Cast Shadow", "Simulate Physics",
    "Mass In Kg", "Gravity Enabled", "Search Details", "Search Outliner",
    "Lit", "Unlit", "Wireframe", "Perspective", "Top", "Front", "Side",
]

UE4 = [
    "Modes", "World Outliner", "Content Browser", "Toolbar", "Details",
    "Place", "Paint", "Landscape", "Foliage", "Geometry Editing",
    "Save Current", "Source Control", "Marketplace", "Launch",
    "Play In Editor", "Standalone Game", "New Editor Window",
    "Build Lighting Only", "Build Geometry", "Build Paths",
]

BLENDER = [
    "File", "Edit", "Render", "Window", "Help", "Add", "Object", "Mesh",
    "Layout", "Modeling", "Sculpting", "UV Editing", "Texture Paint",
    "Shading", "Animation", "Rendering", "Compositing", "Scripting",
    "Scene Collection", "Collection", "Camera", "Light", "Cube", "Sphere",
    "Cylinder", "Plane", "Torus", "Monkey", "Transform", "Dimensions",
    "Delta Transform", "Relations", "Modifier Properties",
    "Object Properties", "Material Properties", "World Properties",
    "Render Properties", "Output Properties", "Particles", "Physics",
    "Constraints", "Data", "Viewport Shading", "Wireframe", "Solid",
    "Material Preview", "Rendered", "Vertex", "Edge", "Face",
    "Subdivision Surface", "Bevel", "Mirror", "Array", "Solidify",
    "Apply", "Shade Smooth", "Shade Flat", "Join", "Separate",
]

BLOCKBENCH = [
    "File", "Edit", "Transform", "Filter", "Display", "View", "Help",
    "New Model", "Open Model", "Save Model", "Export", "Java Block",
    "Bedrock Model", "Modded Entity", "Generic Model", "Skin",
    "Outliner", "Textures", "UV", "Animate", "Paint", "Display Mode",
    "Add Cube", "Add Group", "Add Mesh", "Add Locator", "Add Null Object",
    "Position", "Size", "Pivot", "Inflate", "Rotation", "UV Offset",
    "Texture", "Faces", "North", "South", "East", "West", "Up", "Down",
    "Brush", "Eraser", "Fill", "Color Picker", "Draw Shape", "Gradient",
]

BROWSER = [
    "New Tab", "New Window", "Bookmarks", "History", "Downloads",
    "Settings", "Extensions", "Print", "Find", "Zoom", "Reload",
    "Back", "Forward", "Home", "Search", "Sign in", "Images", "Videos",
    "News", "Maps", "Translate", "Search Google or type a URL",
    "Поиск в Яндексе", "Почта", "Диск", "Карты", "Маркет", "Новости",
    "Войти", "Найти", "Настройки", "Закладки", "История", "Загрузки",
]

EXPLORER = [
    "Home", "Gallery", "Desktop", "Downloads", "Documents", "Pictures",
    "Music", "Videos", "This PC", "Network", "Recycle Bin",
    "Local Disk (C:)", "New", "Cut", "Copy", "Paste", "Rename", "Share",
    "Delete", "Sort", "View", "Filter", "Details", "Properties",
    "Name", "Date modified", "Type", "Size", "Search",
    "Рабочий стол", "Загрузки", "Документы", "Изображения", "Музыка",
    "Видео", "Этот компьютер", "Имя", "Дата изменения", "Тип", "Размер",
    "Создать", "Вырезать", "Копировать", "Вставить", "Переименовать",
    "Удалить", "Свойства", "Вид", "Поиск",
]

APPS = {
    "ue5": UE5, "ue4": UE4, "blender": BLENDER,
    "blockbench": BLOCKBENCH, "browser": BROWSER, "explorer": EXPLORER,
}

# -- разделение словаря ----------------------------------------------------
# Каждая пятая надпись откладывается в проверку и НИКОГДА не показывается
# при обучении. Без этого проверка спрашивает ровно то, что показывала:
# в прошлом прогоне 93% проверочных строк дословно встречались в обучении,
# потеря упала до 0.0067, а «98.5% точности» означали лишь заученный
# список из 254 слов. Читать по буквам модель при этом не научилась.
_ALL = sorted({t for lst in APPS.values() for t in lst})
HELD_OUT = [t for i, t in enumerate(_ALL) if i % 5 == 2]
TRAIN_WORDS = [t for t in _ALL if t not in set(HELD_OUT)]

APPS_TRAIN = {k: [t for t in v if t in set(TRAIN_WORDS)]
              for k, v in APPS.items()}
APPS_TEST = {k: [t for t in v if t in set(HELD_OUT)]
             for k, v in APPS.items()}

# Классы головы «узнать подпись целиком» — только обучающая часть.
LABELS = TRAIN_WORDS
LABEL_ID = {t: i for i, t in enumerate(LABELS)}
N_LABELS = len(LABELS)
