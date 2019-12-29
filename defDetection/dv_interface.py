from dv_end_plane import EndPlane
from dv_sealing_ring import SealingRing
from dv_fillet import Fillet
from dv_inner_plane import InnerPlane
from dv_outer_plane import OuterPlane
from dv_concave_vex_piece import ConcaveVexPiece


class RunTime(object):

    def __init__(self):
        self._ep = EndPlane()
        self._sr = SealingRing()
        self._fl = Fillet()
        self._ip = InnerPlane()
        self._op = OuterPlane()
        self._cp = ConcaveVexPiece()

    def run(self, src, mode):

        if mode == "end_plane":
            self._ep.endPlaneDetection(src)
        elif mode == "sealing_ring":
            self._sr.sealingRingDetection(src)
        elif mode == "fillet":
            self._fl.filletDetection(src)
        elif mode == "inner_plane":
            self._ip.innerPlaneDetection(src)
        elif mode == "outer_plane":
            self._op.outerPlaneDetection(src)
        elif mode == "concave_vex":
            self._cp.concaveVexPieceDetection(src)
