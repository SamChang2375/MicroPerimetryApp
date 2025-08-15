from enum import Enum, auto

class MouseStatus(Enum):
    IDLE = auto()
    DRAW_SEG = auto()
    DRAW_PTS = auto()
    DEL_STR = auto()