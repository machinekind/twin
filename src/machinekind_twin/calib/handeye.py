"""Eye-to-hand calibration by reprojection: cameras fixed, tags on the arm.

Przeniesione z galaxeo-manipulators `sim/calib/handeye.py` (commit 02641c4)
i rozszerzone z jednej kamery na wiele. Matematyka jest ta sama: Levenberg-
Marquardt na bledzie reprojekcji kazdego rogu kazdego taga w kazdym kadrze,
start z zamknietego AX = XB (Park i Martin) albo z pozy nominalnej, bramki na
residuum i na rozrzut obrotow. Doszly dwie rzeczy:

* **Wiele kamer, jedna karta.** Karta jedzie w dloni i jest ta sama dla
  wszystkich kamer, wiec jej poza w chwytaku jest wspolnym parametrem. Kamera,
  ktora widzi karte dobrze, pomaga wtedy tej, ktora widzi ja slabiej: zamiast
  C osobnych problemow po 12 stopni swobody jest jeden o 6C + 6.
* **Residua liczone wektorowo.** Jedno `projectPoints` na kamere zamiast na
  obserwacje - przy kilku kamerach i ~100 obserwacjach petla pythonowa byla
  waskim gardlem calego dopasowania.

    from machinekind_twin.calib.handeye import Observation, solve
    fit = solve(obs, {"front": (K, dist)}, tag_size, {"card": T_card2tcp_nominal})
    fit.cameras["front"].T_cam2base, fit.mounts["card"], fit.rms_px

Every observation is one tag seen in one image while the arm stood still:

    Observation(T_frame2base, T_tag2mount, corners, tag_id, mount, camera)

`T_frame2base` is the pose of the body the mount rides (from forward kinematics
at the measured joint angles), `T_tag2mount` the tag's pose *within* the mount,
which is exact printed geometry, and `corners` the four detected pixel corners.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np

from .tags import corners_in_tag, tag_pose


@dataclass
class Observation:
    T_frame2base: np.ndarray
    T_tag2mount: np.ndarray
    corners: np.ndarray
    tag_id: int = -1
    mount: str = "card"
    camera: str = "cam"
    #: Numer pozy fali, z ktorej jest ta obserwacja (-1 = nieznany) - do diagnostyki.
    pose: int = -1
    #: The measured joint angles [app units] `T_frame2base` came from, for `solve_offsets`.
    joints: Mapping[str, float] | None = None


@dataclass
class CameraFit:
    """Wynik dla jednej kamery."""

    T_cam2base: np.ndarray
    rms_px: float
    max_px: float
    n_obs: int
    #: Rozrzut orientacji karty widzianej przez te kamere, patrz `rotation_spread_R`.
    spread_deg: float
    trusted: bool = False
    reason: str = ""
    #: Residuum kazdej obserwacji tej kamery [px, rms po 4 rogach] i (poza, tag) obok - diagnostyka.
    per_obs_px: list[float] = field(default_factory=list)
    obs_ids: list[tuple[int, int]] = field(default_factory=list)
    #: Obserwacje odrzucone jako odstajace (kadr w ruchu, zla detekcja) przed koncowym dopasowaniem.
    dropped: int = 0


@dataclass
class Fit:
    cameras: dict[str, CameraFit]
    mounts: dict[str, np.ndarray] = field(default_factory=dict)   # mount name -> T_mount2frame
    rms_px: float = float("nan")
    max_px: float = float("nan")
    n_obs: int = 0
    init: str = ""                                                 # which starting point won
    dropped: int = 0
    #: Servo zero offsets fitted with the cameras [app units], on top of those the arm already had:
    #: the model angle is the servo's reading plus the offset. Empty = not fitted or not kept.
    joint_offsets: dict[str, float] = field(default_factory=dict)
    #: What fitting the offsets gave, for the panel: the residual before and after, or why they were left out.
    offsets_note: str = ""

    @property
    def trusted(self) -> bool:
        return bool(self.cameras) and all(c.trusted for c in self.cameras.values())


# ------------------------------------------------------------ SE(3) helpers
def _se3(xi):
    """Exponential map of a 6-vector (rotation vector, translation) to 4x4."""
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(xi[:3], float))
    T[:3, 3] = xi[3:]
    return T


def _log(T):
    return np.concatenate([cv2.Rodrigues(np.asarray(T)[:3, :3])[0].ravel(), np.asarray(T)[:3, 3]])


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _median_R(Rs, R_ref):
    """Mediana obrotow w przestrzeni stycznej w `R_ref`: R_ref @ exp(median(log(R_ref^T R_i)))."""
    rv = np.array([cv2.Rodrigues(R_ref.T @ R)[0].ravel() for R in Rs])
    dR, _ = cv2.Rodrigues(np.median(rv, axis=0))
    return R_ref @ dR


def _median_T(Ts):
    """Elementwise median of a list of transforms, re-orthonormalised.

    Mediana obrotow NIE jest liczona po wektorach Rodriguesa w ukladzie
    globalnym: dla obrotu blisko 180 st. (kamera z boku podstawy) kolejne
    przyblizenia wypadaja po obu stronach zawiniecia +-pi (wektory ~ +v i ~ -v)
    i ich mediana to obrot odlegly o ~180 st. od kazdego z nich. Zamiast tego
    mediana wektorow obrotu wzgledem jednego z wejsc (tam katy sa male, z dala
    od zawiniecia), a potem drugi raz wzgledem wyniku - na wypadek, gdyby
    pierwsze wejscie bylo odstajace."""
    Ts = [np.asarray(T, float) for T in Ts]
    t = np.median([T[:3, 3] for T in Ts], axis=0)
    Rs = [T[:3, :3] for T in Ts]
    R = _median_R(Rs, _median_R(Rs, Rs[0]))
    U, _, Vt = np.linalg.svd(R)                        # usuwa dryf numeryczny z iloczynow
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = U @ Vt, t
    return out


def rotation_spread_R(Rs):
    """How much a set of orientations turned, in the worst direction, in degrees.

    Hand-eye is only conditioned if the carrier rotated about several
    non-parallel axes: a wave that only yaws leaves the camera pose free to
    slide along that axis, and the fit comes back confident and wrong. Stack
    the rotation vectors of every pair of orientations, take the smallest
    singular value of the (3, N) matrix normalised by sqrt(N), and you get the
    rotation available about the least-covered axis. Turning about one axis
    only scores 0, however far it turns."""
    Rs = list(Rs)
    if len(Rs) < 2:
        return 0.0
    v = [cv2.Rodrigues(Rs[i] @ Rs[j].T)[0].ravel() for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    M = np.array(v).T                                  # (3, N) rotation vectors
    # eigenvalues of M M^T rather than the SVD of M: with fewer than three
    # pairs the SVD returns fewer than three singular values and the smallest
    # of them is not the smallest axis.
    lam = np.linalg.eigvalsh(M @ M.T / M.shape[1])
    return float(np.degrees(np.sqrt(max(lam[0], 0.0))))


def rotation_spread(obs):
    """`rotation_spread_R` over the distinct frames of a list of observations."""
    Rs, seen = [], set()
    for o in obs:
        key = o.T_frame2base.tobytes()
        if key not in seen:                     # dwa tagi z jednego kadru to jedna orientacja
            seen.add(key)
            Rs.append(o.T_frame2base[:3, :3])
    return rotation_spread_R(Rs)


# ---------------------------------------------------------------- residuals
class _Problem:
    """Obserwacje ulozone w tablice raz, zeby kazde residuum bylo kilkoma mnozeniami."""

    def __init__(self, obs, cameras, tag_size, names_cam, names_mount):
        self.names_cam, self.names_mount = names_cam, names_mount
        X = np.c_[corners_in_tag(tag_size), np.ones(4)].T            # (4 wsp., 4 rogi)
        self.A = np.array([o.T_frame2base for o in obs])              # (N, 4, 4)
        self.B = np.array([o.T_tag2mount @ X for o in obs])           # (N, 4, 4) rogi w montazu
        self.cam = np.array([names_cam.index(o.camera) for o in obs])
        self.mount = np.array([names_mount.index(o.mount) for o in obs])
        self.corners = np.array([np.asarray(o.corners, float) for o in obs])   # (N, 4, 2)
        self.K = [np.asarray(cameras[c][0], float) for c in names_cam]
        self.dist = [np.zeros(5) if cameras[c][1] is None else np.asarray(cameras[c][1], float)
                     for c in names_cam]

    def unpack(self, p):
        nc = len(self.names_cam)
        cams = [_se3(p[6 * k:6 * k + 6]) for k in range(nc)]
        mounts = [_se3(p[6 * (nc + k):6 * (nc + k) + 6]) for k in range(len(self.names_mount))]
        return cams, mounts

    def residuals(self, p):
        cams, mounts = self.unpack(p)
        M = np.array(mounts)[self.mount]                              # (N, 4, 4)
        P = np.einsum("nij,njk,nkl->nil", self.A, M, self.B)          # rogi w ukladzie podstawy
        out = np.empty((len(self.A), 4, 2))
        for k, T_cam2base in enumerate(cams):
            sel = self.cam == k
            if not sel.any():
                continue
            pts = np.einsum("ij,njl->nil", _inv(T_cam2base), P[sel])[:, :3, :]   # (n, 3, 4)
            pts = pts.transpose(0, 2, 1).reshape(-1, 3)
            px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K[k], self.dist[k])
            out[sel] = px.reshape(-1, 4, 2)
        return (out - self.corners).ravel()


def _params(cams, mounts):
    return np.concatenate([_log(T) for T in list(cams) + list(mounts)])


# ------------------------------------------------------------------- starts
def mount_poses_in_cam(obs, K, dist, tag_size):
    """Per-observation PnP, lifted from the tag to its mount frame."""
    return [tag_pose(o.corners, tag_size, K, dist) @ _inv(o.T_tag2mount) for o in obs]


MIN_PAIR_DEG = 10.0            # relative rotations smaller than this carry no information


def solve_axxb(A, B):
    """Closed-form AX = XB (Park and Martin), the classic hand-eye step.

    `cv2.calibrateHandEye` is gone in OpenCV 5, so this is the same method
    written out. Rotation first: for A X = X B the rotation vectors obey
    R_X log(R_B) = log(R_A), which is an orthogonal Procrustes problem over
    every motion pair. Translation follows linearly from
    (R_A - I) t_X = R_X t_B - t_A, stacked and solved in least squares."""
    a = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in A])
    b = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel() for T in B])
    U, _, Vt = np.linalg.svd(a.T @ b)
    R_X = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    M = np.vstack([TA[:3, :3] - np.eye(3) for TA in A])
    rhs = np.concatenate([R_X @ TB[:3, 3] - TA[:3, 3] for TA, TB in zip(A, B)])
    t_X = np.linalg.lstsq(M, rhs, rcond=None)[0]
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R_X, t_X
    return X


def handeye_init(obs, K, dist, tag_size, names, max_pairs=400):
    """Closed-form start for ONE camera, in the eye-to-hand arrangement.

    The camera is fixed in the base and the card rides the gripper, so what is
    constant across the wave is the card's pose in the gripper:

        T_card2gripper = T_base2gripper_i . T_cam2base . T_card2cam_i

    Equate two poses i and j and the unknowns separate into the standard form
    A X = X B with X = T_cam2base, A = FK_j FK_i^-1 the arm's relative motion
    and B = T_card2cam_j T_card2cam_i^-1 the card's motion as the camera saw
    it. The card's pose in the gripper then follows one per observation, and
    their median is the start."""
    T_m2c = mount_poses_in_cam(obs, K, dist, tag_size)
    pairs = [(i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))]
    A, B = [], []
    for i, j in pairs:
        a = obs[j].T_frame2base @ _inv(obs[i].T_frame2base)
        if np.linalg.norm(cv2.Rodrigues(a[:3, :3])[0]) < np.radians(MIN_PAIR_DEG):
            continue
        A.append(a)
        B.append(T_m2c[j] @ _inv(T_m2c[i]))
        if len(A) >= max_pairs:
            break
    if len(A) < 3:
        raise ValueError("too few distinct rotations for the closed-form hand-eye")
    T_cam2base = solve_axxb(A, B)
    mounts = {}
    for n in names:
        Ts = [_inv(o.T_frame2base) @ T_cam2base @ T for o, T in zip(obs, T_m2c) if o.mount == n]
        if Ts:
            mounts[n] = _median_T(Ts)
    return T_cam2base, mounts


def nominal_init(obs, K, dist, tag_size, mounts_nominal):
    """Start for ONE camera from the rough nominal mount: each observation then
    gives a camera pose, and their median is the start."""
    T_m2c = mount_poses_in_cam(obs, K, dist, tag_size)
    Ts = [o.T_frame2base @ mounts_nominal[o.mount] @ _inv(T) for o, T in zip(obs, T_m2c)]
    return _median_T(Ts), dict(mounts_nominal)


def _lm(p, problem, iters=60):
    lam = 1e-3
    r = problem.residuals(p)
    cost = float(r @ r)
    step = np.zeros_like(p)
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            dp = np.zeros_like(p)
            dp[k] = 1e-6
            J[:, k] = (problem.residuals(p + dp) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            r_new = problem.residuals(p + step)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                p, r, cost, lam, improved = p + step, r_new, c_new, max(lam / 3, 1e-9), True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-10:
            break
    return p, r


def _fit(p, r, problem, obs, tag=""):
    per_corner = np.linalg.norm(r.reshape(-1, 2), axis=1).reshape(-1, 4)   # (N, 4)
    per_obs = np.sqrt(np.mean(per_corner**2, axis=1))                       # (N,)
    cams, mounts = problem.unpack(p)
    out = {}
    for k, name in enumerate(problem.names_cam):
        sel = problem.cam == k
        e = per_corner[sel].ravel()
        mine = [o for o, s in zip(obs, sel) if s]
        out[name] = CameraFit(cams[k], float(np.sqrt(np.mean(e**2))) if e.size else float("inf"),
                              float(e.max()) if e.size else float("inf"), int(sel.sum()),
                              rotation_spread(mine),
                              per_obs_px=[float(v) for v in per_obs[sel]],
                              obs_ids=[(int(o.pose), int(o.tag_id)) for o in mine])
    e = per_corner.ravel()
    return Fit(out, dict(zip(problem.names_mount, mounts)), float(np.sqrt(np.mean(e**2))),
               float(e.max()), len(obs), tag)


def solve(obs, cameras, tag_size, mounts_nominal, iters=60, restarts=5, max_px=1.5, seed=0, min_keep=6):
    """Levenberg-Marquardt on the reprojection error, from the best of several starts.

    `cameras` maps every camera name appearing in `obs` to `(K, dist)`;
    `mounts_nominal` maps every mount name to a rough guess at its pose in the
    frame it rides on. Every camera pose and every mount pose is solved for,
    jointly. Starts: each camera from its own closed-form hand-eye where that
    has enough distinct rotations (the mounts from the median over cameras),
    and everything from the nominal. The lower residual wins; if that is still
    above `max_px` the fit is restarted from jittered versions of it, because
    on a thin session LM can settle in a mirrored minimum with a residual of
    hundreds of pixels."""
    obs = list(obs)
    names_cam = sorted({o.camera for o in obs})
    names_mount = sorted({o.mount for o in obs})
    for group, have in (("camera", cameras), ("mount", mounts_nominal)):
        wanted = names_cam if group == "camera" else names_mount
        missing = [n for n in wanted if n not in have]
        if missing:
            raise ValueError(f"no {group} entry for {missing}")
    problem = _Problem(obs, cameras, tag_size, names_cam, names_mount)

    by_cam = {c: [o for o in obs if o.camera == c] for c in names_cam}
    nominal_cams, eye_cams, eye_mounts = {}, {}, {}
    for c, mine in by_cam.items():
        K, dist = cameras[c]
        nominal_cams[c], _ = nominal_init(mine, K, dist, tag_size, mounts_nominal)
        try:
            T, m = handeye_init(mine, K, dist, tag_size, names_mount)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue                   # too few distinct rotations for the closed form
        eye_cams[c] = T
        for n, Tm in m.items():
            eye_mounts.setdefault(n, []).append(Tm)

    starts = [("nominal", [nominal_cams[c] for c in names_cam],
               [mounts_nominal[n] for n in names_mount])]
    if eye_cams:
        starts.insert(0, ("handeye", [eye_cams.get(c, nominal_cams[c]) for c in names_cam],
                          [_median_T(eye_mounts[n]) if n in eye_mounts else mounts_nominal[n]
                           for n in names_mount]))

    best = None
    for tag, cams0, mounts0 in starts:
        p, r = _lm(_params(cams0, mounts0), problem, iters)
        fit = _fit(p, r, problem, obs, tag)
        if best is None or fit.rms_px < best.rms_px:
            best = fit
    if best.rms_px > max_px:
        rng = np.random.default_rng(seed)
        for i in range(restarts):
            cams0 = []
            for c in names_cam:
                T = best.cameras[c].T_cam2base.copy()
                T[:3, 3] += rng.normal(0, 0.05, 3)
                cams0.append(T)
            p, r = _lm(_params(cams0, [best.mounts[n] for n in names_mount]), problem, iters)
            fit = _fit(p, r, problem, obs, f"jitter{i}")
            if fit.rms_px < best.rms_px:
                best = fit
            if best.rms_px <= max_px:
                break
    return _drop_outliers(best, obs, cameras, tag_size, names_cam, names_mount, iters, max_px, min_keep)


def _drop_outliers(best, obs, cameras, tag_size, names_cam, names_mount, iters, max_px, min_keep=6):
    """Odstajace obserwacje (kadr zlapany w ruchu, zla detekcja, karta drgajaca na uchwycie)
    psuja residuum calej kamery: fala z 20 dobrymi kadrami i 3 zlymi wychodzila 6,9 px.
    Prog: 3 x mediana, nie mniej niz 2 x `max_px`. Odrzucamy najwyzej 30 % obserwacji kamery
    i tylko, gdy zostaje ich co najmniej `min_keep` (sesja podaje swoje `min_obs`: kamera z 14
    kadrami i jednym odstajacym zostawala z 13 i przepadala); dopasowanie od nowa."""
    per_obs = np.empty(len(obs))
    for k, name in enumerate(names_cam):
        sel = np.array([o.camera == name for o in obs])
        per_obs[sel] = best.cameras[name].per_obs_px
    keep = np.ones(len(obs), bool)
    dropped = {}
    for name in names_cam:
        sel = np.array([o.camera == name for o in obs])
        e = per_obs[sel]
        keep_at_least = max(6, int(min_keep))
        if e.size <= keep_at_least:
            continue
        thr = max(3.0 * float(np.median(e)), 2.0 * max_px)
        bad = np.where(sel)[0][e > thr]
        limit = min(int(0.3 * e.size), e.size - keep_at_least)
        if bad.size > limit:                      # najgorsze najpierw
            bad = bad[np.argsort(-per_obs[bad])[:limit]]
        if bad.size:
            keep[bad] = False
            dropped[name] = int(bad.size)
    if not dropped:
        return best
    inliers = [o for o, k in zip(obs, keep) if k]
    problem = _Problem(inliers, cameras, tag_size, names_cam, names_mount)
    cams0 = [best.cameras[c].T_cam2base for c in names_cam]
    p, r = _lm(_params(cams0, [best.mounts[n] for n in names_mount]), problem, iters)
    fit = _fit(p, r, problem, inliers, best.init + "+odstajace")
    if fit.rms_px >= best.rms_px:
        return best
    for name, n in dropped.items():
        fit.cameras[name].dropped = n
    fit.dropped = sum(dropped.values())
    return fit


class _OffsetProblem(_Problem):
    """`_Problem` with servo zero offsets [app units] after the camera and mount parameters.

    The arm's pose of every observation comes from forward kinematics `fk` at the measured joints plus
    the offsets, so it is recomputed whenever the offsets change. A weak prior pulls each offset to
    zero: `prior` of offset costs as much as one pixel, which keeps a direction the cameras cannot
    tell apart from drifting, and does nothing to one they can.
    """

    def __init__(self, obs, cameras, tag_size, names_cam, names_mount, fk, names_joint, prior):
        super().__init__(obs, cameras, tag_size, names_cam, names_mount)
        self.fk, self.names_joint, self.prior = fk, list(names_joint), float(prior)
        self.joints = [dict(o.joints) for o in obs]
        self.n_base = 6 * (len(names_cam) + len(names_mount))
        self._at = np.zeros(len(self.names_joint))      # the offsets `self.A` was computed for

    def offsets(self, p) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names_joint, p[self.n_base:])}

    def residuals(self, p):
        d = np.asarray(p[self.n_base:], float)
        if not np.array_equal(d, self._at):
            self.A = np.array([self.fk({**j, **{n: j[n] + v for n, v in zip(self.names_joint, d)}})
                               for j in self.joints])
            self._at = d.copy()
        return np.concatenate([super().residuals(p[:self.n_base]), d / self.prior])


def solve_offsets(fit: Fit, obs, cameras, tag_size, fk: Callable[[Mapping[str, float]], np.ndarray],
                  names_joint, iters=60, prior=2.0) -> Fit:
    """`fit` fitted again together with zero offsets of the joints `names_joint` [app units].

    Starts from `fit` with no offsets, on the observations it kept (`CameraFit.obs_ids`), each of which
    needs its measured `joints`. `fk` is the forward kinematics the observations' arm poses came from.
    Which joints the cameras can tell apart is the caller's call: an offset of the base's turn is the
    same as every camera standing elsewhere, one of the roll in front of the card the same as the
    card sitting turned in the jaws. The result has `joint_offsets` set; the verdicts are not judged.
    """
    names_cam, names_mount = sorted(fit.cameras), sorted(fit.mounts)
    kept = {(c, int(pose), int(tag)) for c, cf in fit.cameras.items() for pose, tag in cf.obs_ids}
    inliers = [o for o in obs if (o.camera, int(o.pose), int(o.tag_id)) in kept]
    if any(o.joints is None for o in inliers):
        raise ValueError("observations without measured joints - the offsets need them")
    problem = _OffsetProblem(inliers, cameras, tag_size, names_cam, names_mount, fk, names_joint, prior)
    p0 = np.concatenate([_params([fit.cameras[c].T_cam2base for c in names_cam],
                                 [fit.mounts[n] for n in names_mount]), np.zeros(len(problem.names_joint))])
    p, r = _lm(p0, problem, iters)
    out = _fit(p, r[:problem.A.shape[0] * 8], problem, inliers, fit.init + "+zera")
    for name, cam in out.cameras.items():
        cam.dropped = fit.cameras[name].dropped
    out.dropped = fit.dropped
    out.joint_offsets = problem.offsets(p)
    return out


def judge(fit, max_px=1.5, min_spread=8.0, min_obs=14):
    """Bramki galaxeo, osobno dla kazdej kamery.

    Kamera nie jest zaufana, gdy residuum jest za duze, gdy karta widziana
    przez nia obracala sie za malo wokol ktorejs osi (dopasowanie jest wtedy
    pewne siebie i bledne), albo gdy obserwacji jest za malo. Werdykt ma powod,
    a nie tylko "nie" - uzytkownik musi wiedziec, co poprawic w nastepnej fali.
    """
    for cam in fit.cameras.values():
        reasons = []
        if cam.rms_px > max_px:
            reasons.append(f"residuum {cam.rms_px:.2f} px > {max_px} px")
        if cam.spread_deg < min_spread:
            reasons.append(f"rozrzut obrotow {cam.spread_deg:.1f} st. < {min_spread} st. (zdegenerowana fala)")
        if cam.n_obs < min_obs:
            reasons.append(f"tylko {cam.n_obs} obserwacji, potrzeba {min_obs}")
        cam.trusted = not reasons
        if cam.dropped:
            reasons.append(f"odrzucono {cam.dropped} odstajacych kadrow")
        cam.reason = "; ".join(reasons)
    return fit


def pose_error(T_a, T_b):
    """(translation error [m], rotation error [rad]) between two transforms."""
    dT = _inv(T_a) @ T_b
    ang = np.linalg.norm(cv2.Rodrigues(dT[:3, :3])[0])
    return float(np.linalg.norm(dT[:3, 3])), float(ang)
