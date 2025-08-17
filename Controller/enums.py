from enum import Enum, auto

# Setting the status of the mouse
class MouseStatus(Enum):
    IDLE = auto()
    DRAW_SEG = auto()
    DRAW_PTS = auto()
    DEL_STR = auto()
    EDIT_SEG = auto()

class ComputeMode(Enum):
    PRE_SEG = auto()
    APP_SEG = auto()