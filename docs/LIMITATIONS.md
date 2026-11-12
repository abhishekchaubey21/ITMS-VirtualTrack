# Experimental Limitations — ITMS-VirtualTrack (Milestone 1 Baseline)

> **SYNTHETIC SIMULATION DISCLAIMER**: `ITMS-VirtualTrack` generates **SYNTHETIC** data and does not represent real railway infrastructure, certified rolling stock dynamics, or operational safety validation.

## Known Design Boundaries & Assumptions

1. **Pure Kinematic Translation vs. Physical Train Dynamics**:
   - The camera translates smoothly at constant speed along `+Y` (`y = v * t`). It does not simulate rail-wheel hunting oscillation, suspension vibration/pitch jitter, bogie bounce, or aerodynamic buffeting, all of which degrade real-world high-speed imaging.
2. **Idealized Pinhole Optics & Shutter**:
   - The virtual Blender camera uses an idealized pinhole projection without real lens distortion, sensor rolling-shutter skew, dynamic range clipping, dirt/rain on the windshield, or real CMOS read noise unless explicitly added later.
   - EEVEE motion blur approximates temporal shutter integration rather than full physical rolling-shutter readout across scanlines.
3. **Straight Flat Track Geometry**:
   - The `250 m` corridor is straight and level without horizontal curvature, cant/superelevation, vertical grades, switches/turnouts, overhead catenary gantries, or passing traffic.
4. **Simulated Visual Defects vs. Engineering Defects**:
   - `A3` (`rail_crack`) is a controlled geometric/material surface visual feature rather than a metallurgical subsurface fatigue crack; `A4` (`rail_misalignment`) is a parametric geometric deviation rather than a load-induced track-buckling simulation.
5. **Synthetic Domain Gap**:
   - Models trained or evaluated solely on procedural Blender EEVEE scenes face a sim-to-real domain gap and must not be deployed on physical railway tracks without real-world validation.
6. **Hardware Resource Constraints (Local Windows Workstation)**:
   - Drive `C:\` has limited free space (~9.1 GB) while `D:\` (workspace drive) has ~160.8 GB free; large datasets, model weights, and renders must remain on `D:\` and avoid heavy caching on `C:\`.

