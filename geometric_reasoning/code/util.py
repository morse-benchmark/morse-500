from manim import *
import random
import os


def fit_to_camera_3d(
    mobj: Mobject, scene: ThreeDScene, xbuffer: float = 0.05, ybuffer: float = 0.05
):
    """
    Scale + shift *mobj* so that its 2-D projection fills the visible frame
    in a ThreeDScene.  *buffer* is the fractional margin (0 → edge-to-edge).

    Works after arbitrary camera rotations because it projects vertices onto
    the camera-frame basis vectors.
    """
    cam = scene.camera  # ThreeDCamera
    R = (
        cam.get_rotation_matrix()
    )  # world→screen rotation :contentReference[oaicite:0]{index=0}

    # screen-right (+x) and screen-up (+y) directions in WORLD coords
    r_hat = R.T @ RIGHT
    u_hat = R.T @ UP

    verts = mobj.get_all_points()
    w_obj = np.ptp(verts @ r_hat)  # width on screen
    h_obj = np.ptp(verts @ u_hat)  # height on screen

    w_avail = config.frame_width * (1 - 2 * xbuffer)
    h_avail = config.frame_height * (1 - 2 * ybuffer)

    mobj.scale(min(w_avail / w_obj, h_avail / h_obj))

    # ── centre the projection ─────────────────────────────────────────
    verts = mobj.get_all_points()  # after scaling
    proj_center = ((verts @ r_hat).mean()) * r_hat + ((verts @ u_hat).mean()) * u_hat
    target_center = getattr(cam, "frame_center", ORIGIN)  # default is origin
    mobj.shift(target_center - proj_center)

    return mobj
