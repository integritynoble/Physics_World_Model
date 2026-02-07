    # ========================================================================
    # CASSI: UPWMI Algorithm 1 - Brain + Agents with Adaptive Beam Search
    # Operator calibration: mask geo (dx, dy, theta) + dispersion direction (phi_d)
    # ========================================================================
    def test_cassi_correction(self) -> Dict[str, Any]:
        """UPWMI Algorithm 1: Brain + Agents for CASSI with Adaptive Beam Search.

        Implements the UPWMI (Unified Physics World Model Intelligence) framework
        for operator mismatch calibration in CASSI (Coded Aperture Snapshot
        Spectral Imaging) using a structured agent architecture:

        Agents:
          - ReconstructionAgent: proxy (fast) and final (high-quality) GAP-TV recon
          - OperatorAgent: adaptive beam search over operator parameter space
            Uses reconstruction-based scoring for accurate discrimination:
              S(psi) = ||y - A_psi(recon(y, psi))||^2
            with warm-started GAP-TV and staged 1D sweeps for efficiency.
          - VerifierAgent: confidence assessment, residual analysis, convergence

        Brain loop (K iterations):
          1. ProxyRecon with current belief psi^(k)
          2. Adaptive beam search:
             (a) Staged 1D sweeps (reconstruction-based, warm-started)
             (b) 4D beam grid around staged best -> score -> beam keep
             (c) Local coordinate-descent refinement
          3. Update belief psi^(k+1) and world model
          4. Verify confidence and check convergence

        Operator parameters (belief state psi):
          - dx, dy:  mask translation (subpixel shifts)
          - theta:   mask rotation (degrees)
          - phi_d:   dispersion direction rotation (degrees)

        Uses KAIST dataset (256x256x28) with Poisson-Gaussian noise.
        Outputs: OperatorSpec_calib.json, BeliefState.json, Report.json
        """
        self.log("\n[CASSI] UPWMI Algorithm 1: Brain + Agents with Adaptive Beam Search")
        self.log("=" * 70)

        import time as _time
        from dataclasses import dataclass, field as dc_field
        from pwm_core.data.loaders.kaist import KAISTDataset

        # ================================================================
        # Data Structures
        # ================================================================
        @dataclass
        class OperatorSpec:
            """Operator belief psi = (dx, dy, theta, phi_d)."""
            dx: float = 0.0
            dy: float = 0.0
            theta: float = 0.0     # mask rotation (degrees)
            phi_d: float = 0.0     # dispersion direction rotation (degrees)

            def as_dict(self):
                return {"dx": self.dx, "dy": self.dy,
                        "theta_deg": self.theta, "phi_d_deg": self.phi_d}

            def distance(self, other):
                return float(np.sqrt(
                    (self.dx - other.dx) ** 2 + (self.dy - other.dy) ** 2
                    + (self.theta - other.theta) ** 2
                    + (self.phi_d - other.phi_d) ** 2))

            def copy(self):
                return OperatorSpec(self.dx, self.dy, self.theta, self.phi_d)

            def __repr__(self):
                return (f"psi(dx={self.dx:.4f}, dy={self.dy:.4f}, "
                        f"theta={self.theta:.4f}, phi_d={self.phi_d:.4f})")

        @dataclass
        class WorldModel:
            """Full UPWMI world model state."""
            operator_belief: Any          # OperatorSpec
            proxy_ref: Any = None         # SceneBelief.proxy_ref
            final_ref: Any = None         # SceneBelief.final_ref
            verification: Any = None
            decision_log: list = dc_field(default_factory=list)
            psi_trajectory: list = dc_field(default_factory=list)

        # ================================================================
        # Simulation Infrastructure
        # ================================================================
        class _AffineParams:
            __slots__ = ("dx", "dy", "theta_deg")
            def __init__(self, dx=0.0, dy=0.0, theta_deg=0.0):
                self.dx = float(dx)
                self.dy = float(dy)
                self.theta_deg = float(theta_deg)

        def _warp_mask2d(mask2d, affine):
            """Subpixel shift + small rotation via scipy affine_transform."""
            from scipy.ndimage import affine_transform as _at
            H, W = mask2d.shape
            theta = np.deg2rad(affine.theta_deg)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            center = np.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float32)
            M = R.T
            shift = np.array([affine.dy, affine.dx], dtype=np.float32)
            offset = (center - shift) - M @ center
            warped = _at(mask2d.astype(np.float32), matrix=M, offset=offset,
                         output_shape=(H, W), order=1, mode="constant", cval=0.0)
            return np.clip(warped, 0.0, 1.0).astype(np.float32)

        def _make_dispersion_offsets(s_nom, dir_rot_deg):
            theta = np.deg2rad(dir_rot_deg)
            c, s = np.cos(theta), np.sin(theta)
            s_f = s_nom.astype(np.float32)
            return s_f * c, s_f * s

        def _cassi_forward(x_hwl, mask2d, s_nom, dir_rot_deg):
            """CASSI forward model: masked bands placed on expanded canvas."""
            H, W, L = x_hwl.shape
            dx_f, dy_f = _make_dispersion_offsets(s_nom, dir_rot_deg)
            dx_i = np.rint(dx_f).astype(np.int32)
            dy_i = np.rint(dy_f).astype(np.int32)
            if dx_i.min() < 0:
                dx_i = dx_i - int(dx_i.min())
            if dy_i.min() < 0:
                dy_i = dy_i - int(dy_i.min())
            Wp = W + int(dx_i.max())
            Hp = H + int(dy_i.max())
            y = np.zeros((Hp, Wp), dtype=np.float32)
            for l in range(L):
                oy, ox = int(dy_i[l]), int(dx_i[l])
                y[oy:oy + H, ox:ox + W] += mask2d * x_hwl[:, :, l]
            return y

        def _simulate_measurement(cube, mask2d_nom, s_nom, psi, alpha, sigma, rng):
            aff = _AffineParams(psi.dx, psi.dy, psi.theta)
            mask2d_used = _warp_mask2d(mask2d_nom, aff)
            y_clean = _cassi_forward(cube, mask2d_used, s_nom, psi.phi_d)
            y_clean = np.maximum(y_clean, 0.0)
            lam = np.clip(alpha * y_clean, 0.0, 1e9)
            y = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
            y += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)
            return y, mask2d_used

        def _gap_tv_recon(y, cube_shape, mask2d, s_nom, dir_rot_deg,
                          max_iter=80, lam=1.0, tv_weight=0.4, tv_iter=5,
                          x_init=None, gauss_sigma=0.5):
            """GAP-TV reconstruction for expanded-canvas CASSI forward model.

            Args:
                x_init: Optional warm-start initialization (H, W, L) array.
                        If None, uses adjoint initialization.
                gauss_sigma: Gaussian filter sigma for regularization.
                    Use 0.5 for high-quality reconstruction.
                    Use 1.0 for scoring (sharper score landscape for operator search).
            """
            try:
                from skimage.restoration import denoise_tv_chambolle
            except ImportError:
                denoise_tv_chambolle = None
            H, W, L = cube_shape
            dx_f, dy_f = _make_dispersion_offsets(s_nom, dir_rot_deg)
            dx_i = np.rint(dx_f).astype(np.int32)
            dy_i = np.rint(dy_f).astype(np.int32)
            if dx_i.min() < 0:
                dx_i = dx_i - int(dx_i.min())
            if dy_i.min() < 0:
                dy_i = dy_i - int(dy_i.min())
            Wp = W + int(dx_i.max())
            Hp = H + int(dy_i.max())
            # Pad / crop y to canvas
            y_pad = np.zeros((Hp, Wp), dtype=np.float32)
            hh, ww = min(Hp, y.shape[0]), min(Wp, y.shape[1])
            y_pad[:hh, :ww] = y[:hh, :ww]
            y_w = y_pad
            # Phi_sum on canvas
            Phi_sum = np.zeros((Hp, Wp), dtype=np.float32)
            for l in range(L):
                oy, ox = int(dy_i[l]), int(dx_i[l])
                Phi_sum[oy:oy + H, ox:ox + W] += mask2d
            Phi_sum = np.maximum(Phi_sum, 1.0)

            def _A_fwd(x_hwl):
                return _cassi_forward(x_hwl, mask2d, s_nom, dir_rot_deg)

            def _A_adj(r_hw):
                x = np.zeros((H, W, L), dtype=np.float32)
                for l in range(L):
                    oy, ox = int(dy_i[l]), int(dx_i[l])
                    x[:, :, l] += r_hw[oy:oy + H, ox:ox + W] * mask2d
                return x

            if x_init is not None:
                x = x_init.copy()
            else:
                x = _A_adj(y_w / Phi_sum)
            y1 = y_w.copy()
            for _ in range(max_iter):
                yb = _A_fwd(x)
                y1 = y1 + (y_w - yb)
                x = x + lam * _A_adj((y1 - yb) / Phi_sum)
                if denoise_tv_chambolle is not None:
                    for l in range(L):
                        x[:, :, l] = denoise_tv_chambolle(
                            x[:, :, l], weight=tv_weight, max_num_iter=tv_iter)
                else:
                    from scipy.ndimage import gaussian_filter
                    for l in range(L):
                        x[:, :, l] = gaussian_filter(x[:, :, l], sigma=gauss_sigma)
                x = np.clip(x, 0, 1)
            return x.astype(np.float32)

        # ================================================================
        # Agent: Reconstruction
        # ================================================================
        class ReconstructionAgent:
            """Handles proxy (fast) and final (high-quality) reconstruction."""

            def __init__(self, y, mask2d_nom, s_nom, cube_shape):
                self.y = y
                self.mask2d_nom = mask2d_nom
                self.s_nom = s_nom
                self.cube_shape = cube_shape

            def proxy_recon(self, psi, iters=40, x_init=None):
                """Fast proxy reconstruction for operator search iterations."""
                aff = _AffineParams(psi.dx, psi.dy, psi.theta)
                mask = _warp_mask2d(self.mask2d_nom, aff)
                return _gap_tv_recon(self.y, self.cube_shape, mask,
                                     self.s_nom, psi.phi_d, max_iter=iters,
                                     x_init=x_init)

            def final_recon(self, psi, iters=120):
                """High-quality final reconstruction with calibrated operator."""
                aff = _AffineParams(psi.dx, psi.dy, psi.theta)
                mask = _warp_mask2d(self.mask2d_nom, aff)
                return _gap_tv_recon(self.y, self.cube_shape, mask,
                                     self.s_nom, psi.phi_d, max_iter=iters)

        # ================================================================
        # Agent: Operator (Adaptive Beam Search)
        # ================================================================
        class OperatorAgent:
            """Operator calibration via adaptive beam search.

            Uses RECONSTRUCTION-BASED scoring for accurate discrimination:
              S(psi) = ||y - A_psi(recon(y, psi, N_iters))||^2

            Unlike forward-only scoring S(psi; x_ref) = ||y - A_psi(x_ref)||^2
            which is biased toward the reconstruction parameters, reconstruction-
            based scoring is unbiased because each candidate is evaluated with
            its own reconstruction.

            Efficiency: staged 1D sweeps + warm-starting reduces the number of
            full reconstructions needed from O(N^4) to O(N).
            """

            def __init__(self, y, mask2d_nom, s_nom, cube_shape, ranges,
                         beam_width=10, score_iters=15):
                self.y = y
                self.mask2d_nom = mask2d_nom
                self.s_nom = s_nom
                self.cube_shape = cube_shape
                self.ranges = ranges
                self.beam_width = beam_width
                self.score_iters = score_iters
                self._eval_count = 0

            def score_recon(self, psi, x_warm=None, iters=None,
                            gauss_sigma=0.5):
                """Reconstruction-based scoring: reconstruct + residual.

                S(psi) = ||y - A_psi(GAP_TV(y, psi, iters))||^2

                The gauss_sigma parameter controls regularization strength:
                  - sigma=0.5: fine spatial detail, good for dx/phi_d scoring
                  - sigma=1.0: strong regularization, prevents GAP-TV from
                    absorbing dy/theta errors (sharper landscape for dy)

                Args:
                    psi: operator parameters to evaluate
                    x_warm: warm-start initialization (speeds up convergence)
                    iters: reconstruction iterations (default: self.score_iters)
                    gauss_sigma: regularization strength for scoring

                Returns:
                    (score, x_recon) tuple
                """
                if iters is None:
                    iters = self.score_iters
                self._eval_count += 1
                aff = _AffineParams(psi.dx, psi.dy, psi.theta)
                mask = _warp_mask2d(self.mask2d_nom, aff)
                x_recon = _gap_tv_recon(
                    self.y, self.cube_shape, mask, self.s_nom, psi.phi_d,
                    max_iter=iters, x_init=x_warm, gauss_sigma=gauss_sigma)
                y_pred = _cassi_forward(x_recon, mask, self.s_nom, psi.phi_d)
                hh = min(self.y.shape[0], y_pred.shape[0])
                ww = min(self.y.shape[1], y_pred.shape[1])
                r = self.y[:hh, :ww] - y_pred[:hh, :ww]
                return float(np.sum(r * r)), x_recon

            def _staged_1d_sweeps(self, center, log_fn=None):
                """Stage 1: Multi-round 1D sweeps with increasing precision.

                Strategy:
                  Round 1: Sweep easy params first (dx, phi_d) with fewer iters.
                  Round 2: Sweep hard params (dy, theta) with more iters.
                  Round 3: Fine-tune all params around best with higher iters.

                Each candidate starts from its own adjoint (x_warm=None) to
                avoid warm-start bias.

                Returns best psi from staged search.
                """
                r = self.ranges
                best = center.copy()

                # Per-parameter regularization strength:
                #   dx, phi_d: sigma=0.5 (needs fine spatial detail)
                #   dy, theta: sigma=1.0 (prevents error absorption)
                param_sigma = {'dx': 0.5, 'dy': 1.0, 'theta': 0.8, 'phi_d': 0.5}

                def _sweep_param(psi, param, grid, iters, sigma, log_fn=None):
                    """Sweep single param, return best val/score/recon."""
                    bval = getattr(psi, param)
                    bscore = float('inf')
                    bx = None
                    scores_dbg = []
                    for v in grid:
                        test = psi.copy()
                        setattr(test, param, float(v))
                        s, x_r = self.score_recon(test, x_warm=None,
                                                  iters=iters,
                                                  gauss_sigma=sigma)
                        scores_dbg.append((float(v), s))
                        if s < bscore:
                            bscore = s
                            bval = float(v)
                            bx = x_r
                    if log_fn:
                        ranked = sorted(scores_dbg, key=lambda x: x[1])
                        top3 = ", ".join(f"{v:.3f}:{s:.4f}" for v, s in ranked[:3])
                        log_fn(f"      {param}: best={bval:.4f} "
                               f"score={bscore:.4f} [top3: {top3}]")
                    return bval, bscore, bx

                # Round 1: Easy params coarse (dx, phi_d) - sigma=0.5
                iters_easy = self.score_iters + 3
                for param, npts in [('dx', 11), ('phi_d', 9)]:
                    grid = np.linspace(r[f'{param}_min'], r[f'{param}_max'], npts)
                    bv, bs, bx = _sweep_param(best, param, grid, iters_easy,
                                              param_sigma[param], log_fn)
                    setattr(best, param, bv)

                # Round 2: Hard params coarse (dy, theta) - sigma=1.0/0.8
                iters_hard = self.score_iters + 10
                for param, npts in [('dy', 13), ('theta', 9)]:
                    grid = np.linspace(r[f'{param}_min'], r[f'{param}_max'], npts)
                    bv, bs, bx = _sweep_param(best, param, grid, iters_hard,
                                              param_sigma[param], log_fn)
                    setattr(best, param, bv)

                # Round 3: Fine-tune ALL params around current best
                iters_fine = self.score_iters + 10
                param_nfine = {'dx': 7, 'dy': 9, 'theta': 7, 'phi_d': 7}
                for param in ['dx', 'dy', 'theta', 'phi_d']:
                    rng_full = r[f'{param}_max'] - r[f'{param}_min']
                    halfwin = rng_full / 4.0
                    cur = getattr(best, param)
                    lo = max(r[f'{param}_min'], cur - halfwin)
                    hi = min(r[f'{param}_max'], cur + halfwin)
                    grid = np.linspace(lo, hi, param_nfine[param])
                    bv, bs, bx = _sweep_param(best, param, grid, iters_fine,
                                              param_sigma[param], log_fn)
                    setattr(best, param, bv)

                # Final score with fine iters
                final_s, final_x = self.score_recon(best, x_warm=None,
                                                    iters=iters_fine)
                return best, final_s, final_x

            def _beam_refine_4d(self, center, x_warm, log_fn=None):
                """Stage 2: Small 4D grid around staged best -> beam keep.

                Creates a 3^4 = 81 candidate grid centered on the staged best,
                evaluates with reconstruction-based scoring, keeps top-K beam.
                """
                r = self.ranges

                # Step sizes: ~1/3 of the staged sweep step
                steps = {
                    'dx': (r['dx_max'] - r['dx_min']) / 6 / 2,
                    'dy': (r['dy_max'] - r['dy_min']) / 6 / 2,
                    'theta': (r['theta_max'] - r['theta_min']) / 4 / 2,
                    'phi_d': (r['phi_d_max'] - r['phi_d_min']) / 4 / 2,
                }
                clip = {
                    'dx': (r['dx_min'], r['dx_max']),
                    'dy': (r['dy_min'], r['dy_max']),
                    'theta': (r['theta_min'], r['theta_max']),
                    'phi_d': (r['phi_d_min'], r['phi_d_max']),
                }

                # Generate 3^4 grid
                candidates = []
                for ddx in [-1, 0, 1]:
                    for ddy in [-1, 0, 1]:
                        for dth in [-1, 0, 1]:
                            for dpd in [-1, 0, 1]:
                                psi = OperatorSpec(
                                    dx=float(np.clip(center.dx + ddx * steps['dx'],
                                                     *clip['dx'])),
                                    dy=float(np.clip(center.dy + ddy * steps['dy'],
                                                     *clip['dy'])),
                                    theta=float(np.clip(center.theta + dth * steps['theta'],
                                                        *clip['theta'])),
                                    phi_d=float(np.clip(center.phi_d + dpd * steps['phi_d'],
                                                        *clip['phi_d'])),
                                )
                                candidates.append(psi)

                if log_fn:
                    log_fn(f"      Beam grid: {len(candidates)} candidates")

                # Score all with reconstruction-based scoring (sigma=0.7 compromise)
                scored = []
                for psi in candidates:
                    s, x_r = self.score_recon(psi, x_warm=x_warm,
                                              iters=self.score_iters,
                                              gauss_sigma=0.7)
                    scored.append((psi, s, x_r))
                scored.sort(key=lambda x: x[1])

                beam = scored[:self.beam_width]
                if log_fn:
                    log_fn(f"      Beam keep: top-{len(beam)}, "
                           f"best={beam[0][1]:.4f}, "
                           f"worst={beam[-1][1]:.4f}")

                return beam

            def _local_refine(self, psi, x_warm, n_rounds=4, log_fn=None):
                """Stage 3: Coordinate descent refinement with reconstruction scoring.

                Uses a CONSISTENT sigma=0.7 for all params to ensure score
                comparability. The staged sweeps already handle per-param sigma
                optimization; local refinement polishes all params together.
                """
                r = self.ranges
                refine_sigma = 0.7  # consistent for comparability
                best_psi = psi.copy()
                refine_iters = self.score_iters + 10
                best_score, best_x = self.score_recon(psi, x_warm=x_warm,
                                                      iters=refine_iters,
                                                      gauss_sigma=refine_sigma)
                deltas = {'dx': 0.20, 'dy': 0.40, 'theta': 0.05, 'phi_d': 0.02}
                clip = {
                    'dx': (r['dx_min'], r['dx_max']),
                    'dy': (r['dy_min'], r['dy_max']),
                    'theta': (r['theta_min'], r['theta_max']),
                    'phi_d': (r['phi_d_min'], r['phi_d_max']),
                }

                for rd in range(n_rounds):
                    improved = False
                    for param in ['dx', 'dy', 'theta', 'phi_d']:
                        d = deltas[param] / (1 + rd * 0.5)
                        cur = getattr(best_psi, param)
                        for sign in (-1, 1):
                            v = float(np.clip(cur + sign * d,
                                              *clip[param]))
                            test = best_psi.copy()
                            setattr(test, param, v)
                            s, x_r = self.score_recon(
                                test, x_warm=best_x,
                                iters=refine_iters,
                                gauss_sigma=refine_sigma)
                            if s < best_score:
                                best_score = s
                                best_psi = test
                                best_x = x_r
                                improved = True
                    if not improved:
                        break
                    if log_fn:
                        log_fn(f"      Refine round {rd}: {best_psi}, "
                               f"score={best_score:.4f}")

                return best_psi, best_score, best_x

            def sensitivity_sweep(self, psi, x_warm, n_pts=9):
                """Sweep each parameter for sensitivity curves (for VerifierAgent)."""
                r = self.ranges
                param_sigma = {'dx': 0.5, 'dy': 1.0, 'theta': 0.8, 'phi_d': 0.5}
                param_bounds = {
                    'dx': (r['dx_min'], r['dx_max']),
                    'dy': (r['dy_min'], r['dy_max']),
                    'theta': (r['theta_min'], r['theta_max']),
                    'phi_d': (r['phi_d_min'], r['phi_d_max']),
                }
                curves = {}
                for param in ['dx', 'dy', 'theta', 'phi_d']:
                    lo, hi = param_bounds[param]
                    vals = np.linspace(lo, hi, n_pts)
                    curve = []
                    for v in vals:
                        test = psi.copy()
                        setattr(test, param, float(v))
                        s, _ = self.score_recon(test, x_warm=x_warm,
                                                iters=self.score_iters,
                                                gauss_sigma=param_sigma[param])
                        curve.append((float(v), s))
                    curves[param] = curve
                return curves

            def adaptive_beam_search(self, center, log_fn=None):
                """Full adaptive beam search (Algorithm 1, step b).

                Three stages:
                  (a) Staged 1D sweeps (coarse, warm-started)
                  (b) 4D beam grid + beam keep (fine, reconstruction-based)
                  (c) Local coordinate-descent refinement

                Returns: (best_psi, evidence_dict)
                """
                t0 = _time.time()
                self._eval_count = 0

                # (a) Staged 1D sweeps
                if log_fn:
                    log_fn("      Stage (a): 1D sweeps")
                staged_best, staged_score, x_warm = self._staged_1d_sweeps(
                    center, log_fn=log_fn)
                if log_fn:
                    log_fn(f"      Staged best: {staged_best}, "
                           f"score={staged_score:.4f}")

                # (b) 4D beam grid
                if log_fn:
                    log_fn("      Stage (b): 4D beam grid")
                beam = self._beam_refine_4d(staged_best, x_warm, log_fn=log_fn)
                beam_best_psi, beam_best_score, beam_best_x = beam[0]

                # (c) Local refinement on top beam candidate
                if log_fn:
                    log_fn("      Stage (c): Local refinement")
                final_psi, final_score, final_x = self._local_refine(
                    beam_best_psi, beam_best_x, n_rounds=3, log_fn=log_fn)

                # Runner-up for score gap
                runner_up_score = beam[1][1] if len(beam) > 1 else final_score
                score_gap = ((runner_up_score - final_score)
                             / max(abs(final_score), 1e-10))

                # Sensitivity curves
                sensitivity = self.sensitivity_sweep(final_psi, final_x,
                                                     n_pts=9)

                elapsed = _time.time() - t0
                if log_fn:
                    log_fn(f"      DONE: {final_psi}, score={final_score:.4f}, "
                           f"gap={score_gap:.6f}, "
                           f"evals={self._eval_count} ({elapsed:.1f}s)")

                evidence = {
                    'C_0_size': 24,  # staged sweeps
                    'beam_width': len(beam),
                    'C_1_size': self._eval_count,
                    'best_score': float(final_score),
                    'runner_up_score': float(runner_up_score),
                    'score_gap': float(score_gap),
                    'sensitivity': {
                        k: [(float(v), float(s)) for v, s in c]
                        for k, c in sensitivity.items()},
                    'elapsed_s': float(elapsed),
                    'eval_count': self._eval_count,
                }
                return final_psi, evidence

        # ================================================================
        # Agent: Verifier (confidence + stopping)
        # ================================================================
        class VerifierAgent:
            """Assesses per-parameter confidence and convergence.

            Uses:
              - Score gap (best vs runner-up) for overall sharpness
              - Sensitivity curve curvature for per-param confidence
              - Residual MAD for noise sigma estimation
              - psi change norm for convergence
            """

            def __init__(self, y, mask2d_nom, s_nom, cube_shape,
                         tol=0.05, noise_ranges=None):
                self.y = y
                self.mask2d_nom = mask2d_nom
                self.s_nom = s_nom
                self.cube_shape = cube_shape
                self.tol = tol
                self.noise_ranges = noise_ranges or {}

            def verify(self, psi_new, psi_old, evidence, x_ref):
                """Compute confidence for dx, theta, phi_d and verify residual
                structure under OperatorSpec(psi_new).

                Returns dict with: converged, confidence, score_gap,
                psi_change, residual_norm, sigma_hat.
                """
                psi_change = psi_new.distance(psi_old)
                score_gap = evidence.get('score_gap', 0.0)

                # Per-parameter confidence from sensitivity curvature
                confidence = {}
                for param, curve in evidence.get('sensitivity', {}).items():
                    if not curve:
                        confidence[param] = 0.0
                        continue
                    scores = [s for _, s in curve]
                    best_s = min(scores)
                    idx = scores.index(best_s)
                    if 0 < idx < len(scores) - 1:
                        curvature = ((scores[idx - 1] + scores[idx + 1] - 2 * best_s)
                                     / max(best_s, 1e-10))
                        confidence[param] = float(min(1.0, curvature * 50))
                    else:
                        confidence[param] = 0.3

                # Residual analysis -> noise sigma estimate
                aff = _AffineParams(psi_new.dx, psi_new.dy, psi_new.theta)
                mask = _warp_mask2d(self.mask2d_nom, aff)
                y_pred = _cassi_forward(x_ref, mask, self.s_nom, psi_new.phi_d)
                hh = min(self.y.shape[0], y_pred.shape[0])
                ww = min(self.y.shape[1], y_pred.shape[1])
                r = self.y[:hh, :ww] - y_pred[:hh, :ww]
                residual_norm = float(np.sqrt(np.sum(r ** 2)))

                # MAD-based robust sigma estimate
                med = float(np.median(r))
                mad = float(np.median(np.abs(r - med)))
                sigma_hat = mad / 0.6745 if mad > 0 else 0.0
                nr = self.noise_ranges
                if 'sigma_min' in nr and 'sigma_max' in nr:
                    sigma_hat = float(np.clip(
                        sigma_hat, nr['sigma_min'], nr['sigma_max']))

                # Convergence check
                converged = psi_change < self.tol
                high_conf = (all(c > 0.5 for c in confidence.values())
                             if confidence else False)

                return {
                    'converged': converged or (
                        high_conf and psi_change < self.tol * 3),
                    'confidence': confidence,
                    'score_gap': float(score_gap),
                    'psi_change': float(psi_change),
                    'residual_norm': float(residual_norm),
                    'sigma_hat': float(sigma_hat),
                }

        # ================================================================
        # Load Data
        # ================================================================
        dataset = KAISTDataset(resolution=256, num_bands=28)
        name, cube = next(iter(dataset))
        H, W, L = cube.shape
        self.log(f"  Scene: {name} ({H}x{W}x{L})")

        # Nominal 2D mask
        np.random.seed(42)
        mask2d_nom = (np.random.rand(H, W) > 0.5).astype(np.float32)

        # Nominal band shifts (2 pixels per band, matching CASSI step=2)
        s_nom = (np.arange(L, dtype=np.int32) * 2).astype(np.int32)

        # Parameter ranges
        param_ranges = {
            'dx_min': -1.5, 'dx_max': 1.5,
            'dy_min': -1.5, 'dy_max': 1.5,
            'theta_min': -0.30, 'theta_max': 0.30,
            'phi_d_min': -0.12, 'phi_d_max': 0.12,
        }
        noise_ranges = {
            'sigma_min': 0.003, 'sigma_max': 0.015,
            'alpha_min': 600.0, 'alpha_max': 2500.0,
        }

        # ================================================================
        # TRUE parameters (sampled from ranges)
        # ================================================================
        rng = np.random.default_rng(123)
        true_psi = OperatorSpec(
            dx=float(rng.uniform(param_ranges['dx_min'], param_ranges['dx_max'])),
            dy=float(rng.uniform(param_ranges['dy_min'], param_ranges['dy_max'])),
            theta=float(rng.uniform(param_ranges['theta_min'], param_ranges['theta_max'])),
            phi_d=float(rng.uniform(param_ranges['phi_d_min'], param_ranges['phi_d_max'])),
        )
        true_alpha = float(rng.uniform(noise_ranges['alpha_min'], noise_ranges['alpha_max']))
        true_sigma = float(rng.uniform(noise_ranges['sigma_min'], noise_ranges['sigma_max']))
        self.log(f"  TRUE operator: {true_psi}")
        self.log(f"  TRUE noise:    alpha={true_alpha:.0f}, sigma={true_sigma:.4f}")

        # ================================================================
        # Simulate measurement with TRUE operator
        # ================================================================
        y, mask2d_true = _simulate_measurement(
            cube, mask2d_nom, s_nom, true_psi, true_alpha, true_sigma, rng)
        self.log(f"  Measurement shape: {y.shape}")

        # ================================================================
        # Baseline: Reconstruct with WRONG (nominal) params
        # ================================================================
        self.log("\n  [Baseline] Reconstructing with nominal (wrong) params...")
        mask_wrong = _warp_mask2d(mask2d_nom, _AffineParams(0, 0, 0))
        x_wrong = _gap_tv_recon(y, (H, W, L), mask_wrong, s_nom, 0.0,
                                max_iter=80)
        psnr_wrong = compute_psnr(x_wrong, cube)
        self.log(f"  PSNR (wrong):  {psnr_wrong:.2f} dB")

        # ================================================================
        # Baseline: Reconstruct with ORACLE (true) params
        # ================================================================
        self.log("  [Baseline] Reconstructing with oracle (true) params...")
        x_oracle = _gap_tv_recon(y, (H, W, L), mask2d_true, s_nom,
                                 true_psi.phi_d, max_iter=80)
        psnr_oracle = compute_psnr(x_oracle, cube)
        self.log(f"  PSNR (oracle): {psnr_oracle:.2f} dB")

        # ================================================================
        # Algorithm 1, Step 1: Initialize BeliefState & World Model
        # ================================================================
        self.log("\n  === Algorithm 1: UPWMI Brain + Agents ===")
        self.log("  Step 1: Initialize BeliefState with nominal operator")
        psi_0 = OperatorSpec(0.0, 0.0, 0.0, 0.0)
        self.log(f"  psi^(0) = {psi_0}")

        world_model = WorldModel(
            operator_belief=psi_0,
            psi_trajectory=[psi_0],
        )

        # Initialize agents
        recon_agent = ReconstructionAgent(y, mask2d_nom, s_nom, (H, W, L))
        op_agent = OperatorAgent(y, mask2d_nom, s_nom, (H, W, L),
                                 param_ranges, beam_width=10, score_iters=15)
        verifier_agent = VerifierAgent(y, mask2d_nom, s_nom, (H, W, L),
                                       tol=0.05, noise_ranges=noise_ranges)

        # ================================================================
        # Algorithm 1, Step 2: Main Brain Loop
        # ================================================================
        K_max = 3
        t_total = _time.time()
        stop_reason = f"max_iterations_{K_max}"

        for k in range(K_max):
            self.log(f"\n  --- Iteration k={k} " + "-" * 50)
            psi_k = world_model.operator_belief

            # ------------------------------------------------------
            # (a) Reconstruction Agent: proxy recon
            # ------------------------------------------------------
            proxy_iters = 40 + k * 20
            self.log(f"  (a) ReconstructionAgent: ProxyRecon({proxy_iters} iters)")
            self.log(f"      Current belief: {psi_k}")
            t_recon = _time.time()
            x_proxy = recon_agent.proxy_recon(psi_k, iters=proxy_iters)
            world_model.proxy_ref = x_proxy
            self.log(f"      Done in {_time.time() - t_recon:.1f}s")

            # ------------------------------------------------------
            # (b) Operator Agent: adaptive beam search
            # ------------------------------------------------------
            self.log(f"  (b) OperatorAgent: Adaptive beam search")
            psi_star, evidence = op_agent.adaptive_beam_search(
                psi_k, log_fn=self.log)

            # ------------------------------------------------------
            # (c) Operator selection / belief update (brain update)
            # ------------------------------------------------------
            psi_prev = world_model.operator_belief
            world_model.operator_belief = psi_star
            world_model.psi_trajectory.append(psi_star)

            decision_entry = {
                'iteration': k,
                'psi_prev': psi_prev.as_dict(),
                'psi_new': psi_star.as_dict(),
                'C_0_size': evidence['C_0_size'],
                'beam_width': evidence['beam_width'],
                'C_1_size': evidence['C_1_size'],
                'best_score': evidence['best_score'],
                'runner_up_score': evidence['runner_up_score'],
                'score_gap': evidence['score_gap'],
                'elapsed_s': evidence['elapsed_s'],
                'eval_count': evidence.get('eval_count', 0),
            }
            world_model.decision_log.append(decision_entry)
            self.log(f"  (c) Brain update: psi^({k+1}) = {psi_star}")

            # ------------------------------------------------------
            # (d) Verifier Agent: confidence + stopping
            # ------------------------------------------------------
            vr = verifier_agent.verify(psi_star, psi_prev, evidence, x_proxy)
            world_model.verification = vr

            conf_str = ", ".join(
                f"{p}={c:.2f}" for p, c in vr['confidence'].items())
            self.log(f"  (d) VerifierAgent:")
            self.log(f"      delta_psi = {vr['psi_change']:.6f}, "
                     f"score_gap = {vr['score_gap']:.6f}")
            self.log(f"      sigma_hat = {vr['sigma_hat']:.4f}, "
                     f"residual_norm = {vr['residual_norm']:.1f}")
            self.log(f"      confidence: [{conf_str}]")
            self.log(f"      converged = {vr['converged']}")

            if vr['converged'] and k >= 1:
                stop_reason = f"converged_at_k={k}"
                self.log(f"  *** CONVERGED at iteration {k} ***")
                break
            elif vr['converged'] and k < 1:
                self.log(f"  (converged but k={k} < 2, continuing...)")

        loop_time = _time.time() - t_total

        # ================================================================
        # Algorithm 1, Step 3: FinalRecon Agent
        # ================================================================
        psi_final = world_model.operator_belief
        self.log(f"\n  Step 3: FinalRecon Agent")
        self.log(f"  Final belief psi_hat = {psi_final}")
        t_final = _time.time()
        x_final = recon_agent.final_recon(psi_final, iters=120)
        world_model.final_ref = x_final
        final_time = _time.time() - t_final
        self.log(f"  FinalRecon done in {final_time:.1f}s")

        psnr_corrected = compute_psnr(x_final, cube)

        # ================================================================
        # Algorithm 1, Step 4: Outputs (world model artifacts)
        # ================================================================
        self.log(f"\n  Step 4: Generating output artifacts")

        # OperatorSpec_calib.json
        calib = psi_final.as_dict()

        # BeliefState.json
        belief_state = {
            'psi_trajectory': [p.as_dict() for p in world_model.psi_trajectory],
            'stop_reason': stop_reason,
            'total_iterations': len(world_model.decision_log),
            'total_loop_time_s': float(loop_time),
            'decision_log': world_model.decision_log,
        }

        # Report.json
        diagnosis = {
            'dx_error': abs(psi_final.dx - true_psi.dx),
            'dy_error': abs(psi_final.dy - true_psi.dy),
            'theta_error': abs(psi_final.theta - true_psi.theta),
            'phi_d_error': abs(psi_final.phi_d - true_psi.phi_d),
        }
        vr_final = world_model.verification or {}
        report = {
            'diagnosis': diagnosis,
            'confidence': vr_final.get('confidence', {}),
            'sigma_hat': vr_final.get('sigma_hat', None),
            'psnr_wrong': float(psnr_wrong),
            'psnr_corrected': float(psnr_corrected),
            'psnr_oracle': float(psnr_oracle),
            'improvement_db': float(psnr_corrected - psnr_wrong),
        }

        # Save JSON artifacts
        output_dir = Path(__file__).parent / "results" / "cassi_upwmi"
        output_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in [("OperatorSpec_calib.json", calib),
                            ("BeliefState.json", belief_state),
                            ("Report.json", report)]:
            with open(output_dir / fname, "w") as f:
                json.dump(data, f, indent=2, default=str)

        # ================================================================
        # Summary
        # ================================================================
        self.log(f"\n  {'=' * 60}")
        self.log(f"  RESULTS SUMMARY")
        self.log(f"  {'=' * 60}")
        self.log(f"  True operator:       {true_psi}")
        self.log(f"  Calibrated operator: {psi_final}")
        self.log(f"  Errors: dx={diagnosis['dx_error']:.4f}, "
                 f"dy={diagnosis['dy_error']:.4f}, "
                 f"theta={diagnosis['theta_error']:.4f} deg, "
                 f"phi_d={diagnosis['phi_d_error']:.4f} deg")
        self.log(f"  True noise:  alpha={true_alpha:.0f}, sigma={true_sigma:.4f}")
        self.log(f"  Est. sigma:  {vr_final.get('sigma_hat', 'N/A')}")
        self.log(f"  PSNR wrong:     {psnr_wrong:.2f} dB")
        self.log(f"  PSNR corrected: {psnr_corrected:.2f} dB "
                 f"(+{psnr_corrected - psnr_wrong:.2f} dB)")
        self.log(f"  PSNR oracle:    {psnr_oracle:.2f} dB")
        self.log(f"  Stop reason: {stop_reason}")
        self.log(f"  Total time:  {loop_time:.1f}s (loop) + "
                 f"{final_time:.1f}s (final recon)")
        self.log(f"  Artifacts:   {output_dir}")

        result = {
            "modality": "cassi",
            "algorithm": "UPWMI_Algorithm1_AdaptiveBeamSearch",
            "mismatch_param": ["mask_geo", "disp_dir_rot", "noise"],
            "true_value": {
                "geo": {"dx": true_psi.dx, "dy": true_psi.dy,
                        "theta_deg": true_psi.theta},
                "disp": {"dir_rot_deg": true_psi.phi_d},
                "noise": {"alpha": true_alpha, "sigma": true_sigma},
            },
            "wrong_value": {
                "geo": {"dx": 0.0, "dy": 0.0, "theta_deg": 0.0},
                "disp": {"dir_rot_deg": 0.0},
                "noise": {"alpha": None, "sigma": None},
            },
            "calibrated_value": calib,
            "oracle_psnr": float(psnr_oracle),
            "psnr_without_correction": float(psnr_wrong),
            "psnr_with_correction": float(psnr_corrected),
            "improvement_db": float(psnr_corrected - psnr_wrong),
            "belief_state": belief_state,
            "report": report,
        }

        return result
