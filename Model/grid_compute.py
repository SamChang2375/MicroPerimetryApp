# Model/grid_compute.py
from dataclasses import dataclass
from .annotations import AnnotationsBundle

@dataclass(frozen=True)
class GridResult:
    ok: bool
    message: str
    debug: dict

class GridComputer:
    def compute(self, bundle: AnnotationsBundle) -> GridResult:
        # TODO: hier später deine echte Berechnung
        return GridResult(
            ok=True,
            message="Grids computed",
            debug={
                "hr_points": len(bundle.highres.points),
                "hr_seg_len": len(bundle.highres.seg),
                "sd_points": len(bundle.sd.points),
                "sd_seg_len": len(bundle.sd.seg),
                "mp_points": len(bundle.micro.points),
            },
        )