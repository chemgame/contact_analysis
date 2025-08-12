#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import argparse
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import time
import os
import sys
from datetime import datetime
import shutil


# ---- Logging setup (console + logfile; suppress MDAnalysis chatter) ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)
logging.getLogger('MDAnalysis.coordinates').setLevel(logging.WARNING)
logging.getLogger('MDAnalysis.topology').setLevel(logging.WARNING)


# ---- Utilities ----

def check_input_files(top, traj):
    """Verify required inputs exist; exit with error if missing."""
    missing = []
    if not os.path.exists(top):  missing.append(f"Topology file: {top}")
    if not os.path.exists(traj): missing.append(f"Trajectory file: {traj}")
    if missing:
        logger.error("ERROR: Missing input files:")
        for f in missing:
            logger.error(" - %s", f)
        sys.exit(1)
    logger.info("Input files found and validated.")


def backup_file(path):
    """Backup existing file to filename_DDMMYYYY-HHMMSS.ext; return backup path or None."""
    if os.path.exists(path):
        d = os.path.dirname(path)
        name, ext = os.path.splitext(os.path.basename(path))
        stamp = datetime.now().strftime("%d%m%Y-%H%M%S")
        bkp = os.path.join(d, f"{name}_{stamp}{ext}") if d else f"{name}_{stamp}{ext}"
        shutil.copy2(path, bkp)
        logger.info("Backed up existing file: %s -> %s", path, bkp)
        return bkp
    return None


def backup_output_files(paths_dict):
    """Backup all outputs that would be overwritten."""
    logger.info("Checking for existing output files to backup...")
    for _, pth in paths_dict.items():
        backup_file(pth)


def _estimate_dt_ps(times):
    """Estimate dt from a time array if possible; else None (units neglected by request)."""
    t = np.asarray(times, dtype=np.float64)
    if t.size >= 2 and np.all(np.isfinite(t)):
        d = np.diff(t)
        d = d[np.isfinite(d)]
        if d.size:
            return float(np.median(d))
    return None


def _extract_windows(bound, begin, dt_ps, min_dur, gap_fill, times=None):
    """Build bound windows, merge small gaps, drop short runs. Units: treated as-is."""
    b = np.asarray(bound, dtype=bool)
    n = b.size
    if n == 0 or not b.any():
        return []

    x = b.astype(np.int8)
    dx = np.diff(x)
    starts = list(np.where(dx == 1)[0] + 1)
    ends   = list(np.where(dx == -1)[0])
    if b[0]:  starts = [0] + starts
    if b[-1]: ends   = ends + [n - 1]
    runs = [(s, e) for s, e in zip(starts, ends)]

    if dt_ps is None:
        return [dict(start_f=begin+s, end_f=begin+e, start_ps=None, end_ps=None, dur_ps=None) for s, e in runs]

    # Merge gaps ≤ gap_fill
    merged = []
    for s, e in runs:
        if not merged:
            merged.append([s, e]); continue
        ps, pe = merged[-1]
        gap_frames = s - pe - 1
        if gap_frames * dt_ps <= gap_fill:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    kept = []
    for s, e in merged:
        dur_ps = (e - s + 1) * dt_ps
        if dur_ps >= min_dur:
            kept.append((s, e, dur_ps))

    rows = []
    if times is not None and len(times) == n and np.all(np.isfinite(times)):
        t = np.asarray(times, dtype=np.float64)
        for s, e, dur_ps in kept:
            rows.append(dict(start_f=begin+s, end_f=begin+e,
                             start_ps=float(t[s]), end_ps=float(t[e]), dur_ps=float(dur_ps)))
    else:
        for s, e, dur_ps in kept:
            rows.append(dict(start_f=begin+s, end_f=begin+e,
                             start_ps=float(s*dt_ps), end_ps=float(e*dt_ps), dur_ps=float(dur_ps)))
    return rows


def _write_events_dat(rows, path):
    with open(path, 'w') as f:
        f.write("# start_frame\tend_frame\tstart_ps\tend_ps\tduration_ps\n")
        for w in rows:
            f.write(f"{w['start_f']}\t{w['end_f']}\t{w['start_ps']}\t{w['end_ps']}\t{w['dur_ps']}\n")


def _write_dist_traces(t, dmin, dmean, path):
    with open(path, 'w') as f:
        f.write("# time_ps\tmin_dist_A\tmean_dist_A\n")
        for ti, mi, ai in zip(t, dmin, dmean):
            ti_out = f"{ti:.6f}" if np.isfinite(ti) else "NA"
            f.write(f"{ti_out}\t{mi:.6f}\t{ai:.6f}\n")


def residue_min_map(pos1, ridx1, pos2, ridx2, n1, n2, cutoff, box):
    """Per-frame residue×residue capped-min distance (Å). Missing pairs become inf."""
    dmin = np.full((n1, n2), np.inf, dtype=np.float32)
    pairs, d = capped_distance(pos1, pos2, max_cutoff=cutoff, box=box, return_distances=True)
    if pairs.size:
        r1 = ridx1[pairs[:, 0]]
        r2 = ridx2[pairs[:, 1]]
        np.minimum.at(dmin, (r1, r2), d.astype(np.float32))
    return dmin


# ---- Core calculation ----

def calculate_proximity_and_contacts(u, s1, s2, begin=0, end=None, on_thresh=3.5,
                                     off_thresh=6.0, stride=1, trace_true_min=False):
    """
    Heavy-atom only pipeline:
      - atom-level contact probabilities for BOTH groups (hysteresis: ON if min<on_thresh; OFF when missing);
      - residue hits for group-1 (for bar plot; unchanged per request);
      - residue×residue capped-min map and contact frequency;
      - distance traces for group-1 based on capped minima (or true-min with large cap).
    """
    # Enforce heavy-only atoms for ALL calculations
    p = u.select_atoms(f"{s1} and not name H*")
    r = u.select_atoms(f"{s2} and not name H*")

    res1 = list(p.residues)
    res2 = list(r.residues)

    # Residue index maps (heavy-only)
    n1, n2 = len(res1), len(res2)
    skip_map = (n1 <= 1 or n2 <= 1 or len(p) == 0 or len(r) == 0)

    resid_map1 = {rr.resid: i for i, rr in enumerate(res1)}
    resid_map2 = {rr.resid: i for i, rr in enumerate(res2)}
    ridx1 = np.array([resid_map1[a.resid] for a in p.atoms], dtype=np.int32) if len(p) else np.array([], dtype=np.int32)
    ridx2 = np.array([resid_map2[a.resid] for a in r.atoms], dtype=np.int32) if len(r) else np.array([], dtype=np.int32)

    # For group-1 residue bar (kept as before)
    p_residues = res1
    n_res = len(p_residues)

    # Dense maps (kept as-is)
    if not skip_map:
        dsum = np.zeros((n1, n2), dtype=np.float32)
        ccnt = np.zeros((n1, n2), dtype=np.int32)
        cstate = np.zeros((n1, n2), dtype=np.bool_)

    # Atom-level accumulators for BOTH groups
    n_p = len(p)
    n_r = len(r)
    contact_count_p = np.zeros(n_p, dtype=np.int32)
    contact_count_r = np.zeros(n_r, dtype=np.int32)
    cstate_p = np.zeros(n_p, dtype=np.bool_)
    cstate_r = np.zeros(n_r, dtype=np.bool_)
    res_contact_count_p = np.zeros(n_res, dtype=np.int32)  # for group-1 residue bar only

    # Frame range
    if end is None or end == -1:
        end = len(u.trajectory)
    total_frames = (end - begin + stride - 1) // stride
    if total_frames <= 0:
        logger.error("No frames to process (begin >= end or empty trajectory).")
        sys.exit(1)

    logger.info("Total trajectory frames: %d", len(u.trajectory))
    logger.info("Processing %d frames (stride=%d).", total_frames, stride)

    bound_trace, times_ps = [], []
    dmin_trace, dmean_trace = [], []

    # ---- Frame loop ----
    for ts in tqdm(u.trajectory[begin:end:stride], total=total_frames):
        box = ts.dimensions
        p_pos = p.positions
        r_pos = r.positions

        # Per-atom minimal distances with OFF cutoff (capped) for BOTH groups
        pairs, d = capped_distance(p_pos, r_pos, max_cutoff=off_thresh, box=box, return_distances=True)

        # p-side minima
        min_p = np.full(n_p, np.inf, dtype=np.float32)
        if pairs.size:
            np.minimum.at(min_p, pairs[:, 0], d.astype(np.float32))

        # r-side minima
        min_r = np.full(n_r, np.inf, dtype=np.float32)
        if pairs.size:
            np.minimum.at(min_r, pairs[:, 1], d.astype(np.float32))

        # Hysteresis:
        #   ON when min < on_thresh
        #   OFF when missing (non-finite). (Dropped '> off_thresh' per request.)
        on_p  = min_p < on_thresh
        off_p = ~np.isfinite(min_p)
        cstate_p[on_p]  = True
        cstate_p[off_p] = False

        on_r  = min_r < on_thresh
        off_r = ~np.isfinite(min_r)
        cstate_r[on_r]  = True
        cstate_r[off_r] = False

        contact_count_p += cstate_p.astype(np.int32)
        contact_count_r += cstate_r.astype(np.int32)

        # Residue hits for selection-1 (for bar)
        if n_res and n_p:
            hits_p = np.bincount(ridx1[cstate_p], minlength=n_res) > 0
            res_contact_count_p += hits_p.astype(np.int32)

        # Selection-level bound state (symmetric OR)
        bound_trace.append(bool(cstate_p.any() or cstate_r.any()))
        times_ps.append(getattr(ts, "time", np.nan))  # units neglected intentionally

        # Distance traces (group-1): capped or "true-min" via large-cap neighbor search
        trace_md = min_p.copy()
        missing = ~np.isfinite(trace_md)

        if trace_true_min and missing.any():
            if box is not None and np.all(ts.dimensions[:3] > 0):
                a, b, c = ts.dimensions[:3]
                rmax = float(np.linalg.norm(np.array([a, b, c]) * 0.5))
            else:
                all_pos = np.vstack((p_pos, r_pos))
                rmax = float(np.linalg.norm(all_pos.max(0) - all_pos.min(0)))
            lp = p_pos[missing]
            pairs2, d2 = capped_distance(lp, r_pos, max_cutoff=rmax, box=box, return_distances=True)
            if pairs2.size:
                idx_missing = np.where(missing)[0]
                np.minimum.at(trace_md, idx_missing[pairs2[:, 0]], d2.astype(np.float32))
        elif missing.any():
            trace_md[missing] = off_thresh

        finite = np.isfinite(trace_md)
        dmin_trace.append(float(np.min(trace_md[finite])) if finite.any() else float('nan'))
        dmean_trace.append(float(np.mean(trace_md[finite])) if finite.any() else float('nan'))

        # Residue–residue map via neighbor list (heavy-only)
        if not skip_map:
            dmin = residue_min_map(p_pos, ridx1, r_pos, ridx2, n1, n2, off_thresh, box)
            on = dmin < on_thresh
            off = ~np.isfinite(dmin)  # drop (> off_thresh)
            cstate[on]  = True
            cstate[off] = False
            dsum += np.where(np.isfinite(dmin), dmin, off_thresh)  # capped-mean construction
            ccnt += cstate.astype(np.int32)

    frame_count = total_frames
    atom_prob_p = contact_count_p.astype(np.float32) / frame_count
    atom_prob_r = contact_count_r.astype(np.float32) / frame_count
    avg_dist    = dsum / frame_count if not skip_map else None
    contact_freq = ccnt / frame_count if not skip_map else None
    res_prob_p  = res_contact_count_p.astype(np.float32) / frame_count

    return (atom_prob_p, atom_prob_r, avg_dist, contact_freq,
            res1, res2, p_residues, res_prob_p,
            np.asarray(bound_trace, dtype=bool), np.asarray(times_ps, dtype=np.float64),
            np.asarray(dmin_trace, dtype=np.float64), np.asarray(dmean_trace, dtype=np.float64))


# ---- Plotting ----

def plot_residue_bar(residues, probs, out_path):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True, dpi=200)
    res_ids = [res.resid for res in residues]
    ax.bar(res_ids, probs, edgecolor='b', linewidth=0.5, color='b')
    ax.set_xlabel('Protein Residues', fontsize=22)
    ax.set_ylabel('Proximity probability', fontsize=22)
    if len(res_ids) > 0:
        ax.set_xlim(-4, res_ids[-1] + 5)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=18)
    plt.savefig(out_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_map(avg_dist, res1, res2, out_path, xlabel, ylabel, off_thresh, cmap):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=200)
    n1, n2 = avg_dist.shape
    # Perceptually uniform; fixed scale [0, OFF] because we use capped mean
    im = ax.imshow(avg_dist, cmap=cmap, vmin=0.0, vmax=off_thresh,
                   aspect='auto', origin='lower', extent=[1, n2, 1, n1])
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=18)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(f'Capped distance (Å)', fontsize=22)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(out_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_contact_probability_map(contact_freq, res1, res2, out_path, xlabel, ylabel, cmap):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=200)
    n1, n2 = contact_freq.shape
    im = ax.imshow(contact_freq, cmap=cmap, aspect='auto', origin='lower',
                   vmin=0, vmax=1, extent=[1, n2, 1, n1])
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=18)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Contact Probability', fontsize=22)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(out_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


# ---- Reporting / I/O helpers ----

def _log_system_specs(u, p_all, r_all, begin, end, on, off, ev, md, gf, stride):
    """Structured system specs; p_all/r_all are original (may include H) selections for reporting."""
    water = u.select_atoms('resname TIP3 SOL HOH WAT')
    ions  = u.select_atoms('resname NA K CL MG CA MN ZN CU CO FE LI RB CS SR BA F BR I')
    nF = len(u.trajectory)
    dt = getattr(u.trajectory, 'dt', None)
    logger.info("System specs: atoms=%d residues=%d waters=%d ions=%d frames=%d dt(ps)=%s",
                u.atoms.n_atoms, len(u.residues), len(water.residues), len(ions.residues),
                nF, f"{dt:.3f}" if dt is not None else "NA")
    logger.info("Selections: s1 atoms=%d (heavy=%d, residues=%d); s2 atoms=%d (heavy=%d, residues=%d)",
                len(p_all), len(p_all.select_atoms('not name H*')), len(p_all.residues),
                len(r_all), len(r_all.select_atoms('not name H*')), len(r_all.residues))
    if hasattr(u.trajectory.ts, 'dimensions') and u.trajectory.ts.dimensions is not None:
        a, b, c = u.trajectory.ts.dimensions[:3]
        logger.info("Box (Å): a=%.3f b=%.3f c=%.3f", a, b, c)
    logger.info("Frame window: begin=%d end=%s stride=%d", begin, end if end is not None else "last", stride)
    logger.info("Hysteresis: ON=%.2f Å, OFF=%.2f Å", on, off)
    logger.info("Events: file=%s min_dur=%.1f gap_fill=%.1f", ev if ev else "None", md, gf)


def _residue_max_from_atom_probs(atoms, probs):
    """Return mapping resid -> max(atom_prob) for an AtomGroup + per-atom prob array."""
    if len(atoms) == 0:
        return {}
    resid = np.array([a.resid for a in atoms], dtype=np.int32)
    idx = np.argsort(resid, kind='mergesort')
    resid_sorted = resid[idx]
    probs_sorted = np.asarray(probs, dtype=np.float32)[idx]
    uniq, start_idx = np.unique(resid_sorted, return_index=True)
    max_vals = np.maximum.reduceat(probs_sorted, start_idx)
    return {int(r): float(v) for r, v in zip(uniq, max_vals)}


def write_residue_dat(resid_map_p, resid_map_r, psel, rsel, out_path):
    """
    Write residues (both groups) with probabilities (max atom probability per residue).
    Format: 'group<TAB>resname_resid<TAB>probability'
    """
    with open(out_path, 'w') as f:
        f.write("# group\tresname_resid\tprobability\n")
        # Group 1
        rows = []
        for res in psel.residues:
            pr = resid_map_p.get(res.resid, 0.0)
            rows.append(("s1", f"{res.resname}{res.resid}", pr))
        rows.sort(key=lambda x: x[2], reverse=True)
        for g, tag, pr in rows:
            f.write(f"{g}\t{tag}\t{pr:.4f}\n")
        # Group 2
        rows = []
        for res in rsel.residues:
            pr = resid_map_r.get(res.resid, 0.0)
            rows.append(("s2", f"{res.resname}{res.resid}", pr))
        rows.sort(key=lambda x: x[2], reverse=True)
        for g, tag, pr in rows:
            f.write(f"{g}\t{tag}\t{pr:.4f}\n")


# ---- Main ----

def main():
    # Argparse with consistent style (short & long flags; metavar=""; dtype; default/required in help)
    p = argparse.ArgumentParser(
        description="Analyze contacts between two atom groups (heavy atoms only) with hysteresis; outputs PDB, DAT, maps, bar plot; optional event windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g_io     = p.add_argument_group('Inputs')
    g_out    = p.add_argument_group('Outputs')
    g_sel    = p.add_argument_group('Selections')
    g_plot   = p.add_argument_group('Plotting')
    g_frames = p.add_argument_group('Time frames')
    g_thresh = p.add_argument_group('Contact thresholds')
    g_evt    = p.add_argument_group('Event windows')
    g_log    = p.add_argument_group('Logging')

    # Inputs
    g_io.add_argument('-s',  '--topology',   metavar="", type=str, required=True,
                      help='Topology file (str, required)')
    g_io.add_argument('-f',  '--trajectory', metavar="", type=str, required=True,
                      help='Trajectory file (str, required)')

    # Outputs
    g_out.add_argument('-o',  '--output',   metavar="", type=str, default='contact.pdb',
                       help='Output PDB (str, default=contact.pdb)')
    g_out.add_argument('-d',  '--datfile',  metavar="", type=str, default='contact.dat',
                       help='Residue probabilities DAT (str, default=contact.dat)')
    g_out.add_argument('-dm', '--distmap',  metavar="", type=str, default='distance_map.jpg',
                       help='Distance map JPG (str, default=distance_map.jpg)')
    g_out.add_argument('-cm', '--contmap',  metavar="", type=str, default='contact_map.jpg',
                       help='Contact map JPG (str, default=contact_map.jpg)')
    g_out.add_argument('-bp', '--barplot',  metavar="", type=str, default='contact_bar.jpg',
                       help='Residue bar plot JPG (str, default=contact_bar.jpg)')
    g_out.add_argument('-dd', '--distdat',  metavar="", type=str, default='dist_traces.dat',
                       help='Distance traces DAT (str, default=dist_traces.dat)')
    g_out.add_argument('-tm', '--trace_true_min', action='store_true',
                       help='Distance traces use true minima via large-cap neighbor search (bool, default=False)')
    g_out.add_argument('-ev', '--events',   metavar="", type=str, default=None,
                       help='Bound windows output DAT (str, default=None=skip)')

    # Selections
    g_sel.add_argument('-s1', '--selection1', metavar="", type=str, required=True,
                       help='Group-1 selection (str, required)')
    g_sel.add_argument('-s2', '--selection2', metavar="", type=str, required=True,
                       help='Group-2 selection (str, required)')

    # Plotting
    g_plot.add_argument('-x',  '--xlabel',     metavar="", type=str, default='Residue index (group2)',
                       help='X label (str, default=Residue index (group2))')
    g_plot.add_argument('-y',  '--ylabel',     metavar="", type=str, default='Residue index (group1)',
                       help='Y label (str, default=Residue index (group1))')
    g_plot.add_argument('-cmd', '--cmap_dist', metavar="", type=str, default='magma',
                       help='Colormap for distance map (str, default=magma)')
    g_plot.add_argument('-cmc', '--cmap_cont', metavar="", type=str, default='viridis',
                       help='Colormap for contact map (str, default=viridis)')


    # Frames & timing (units intentionally not enforced)
    g_frames.add_argument('-b',  '--begin',  metavar="", type=int,   default=0,
                          help='Begin frame index (int, default=0)')
    g_frames.add_argument('-e',  '--end',    metavar="", type=int,   default=None,
                          help='End frame index (int, default=None=last)')
    g_frames.add_argument('-st', '--stride', metavar="", type=int,   default=1,
                          help='Use every k-th frame (int, default=1)')
    g_frames.add_argument('-dt', '--dt_ps',  metavar="", type=float, default=None,
                          help='Time step in ps if missing (float, default=None)')

    # Contact thresholds
    g_thresh.add_argument('-on',  '--on_thresh',  metavar="", type=float, default=3.5,
                          help='ON threshold in Å (float, default=3.5)')
    g_thresh.add_argument('-off', '--off_thresh', metavar="", type=float, default=6.0,
                          help='OFF threshold in Å (float, default=6.0)')

    # Event windows
    g_evt.add_argument('-md', '--min_dur',  metavar="", type=float, default=50.0,
                       help='Min bound duration in ps (float, default=50.0)')
    g_evt.add_argument('-gf', '--gap_fill', metavar="", type=float, default=20.0,
                       help='Merge gaps ≤ this (ps) (float, default=20.0)')

    # Logging
    g_log.add_argument('-g', '--log', metavar="", type=str, default='contact.log',
                       help='Logfile (str, default=contact.log)')

    args = p.parse_args()

    # Logfile: backup if exists, then attach file handler (console already set)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    if args.log:
        _ = backup_file(args.log)
        fh = logging.FileHandler(args.log, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    # ---- Top-of-log header (command first) ----
    logger.info("Command: %s", " ".join(sys.argv))
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("PROXIMITY AND CONTACT ANALYSIS STARTING")
    logger.info("=" * 80)
    logger.info("Start time: %s", time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(start_time)))
    logger.info("Topology file: %s", args.topology)
    logger.info("Trajectory file: %s", args.trajectory)
    logger.info("Selection 1: %s", args.selection1)
    logger.info("Selection 2: %s", args.selection2)
    logger.info("Frame range: %s to %s; stride=%d", args.begin, args.end if args.end is not None else 'end', args.stride)
    logger.info("Contact thresholds: ON=%.2f Å, OFF=%.2f Å", args.on_thresh, args.off_thresh)
    logger.info("Distance-trace mode: %s",
                "true-min (large-cap NSGrid)" if args.trace_true_min else f"capped @ OFF={args.off_thresh} Å")

    # I/O checks & backups
    check_input_files(args.topology, args.trajectory)
    out_paths = {
        "PDB": args.output,
        "DAT": args.datfile,
        "Distance map": args.distmap,
        "Contact map": args.contmap,
        "Bar plot": args.barplot,
        "Distance traces DAT": args.distdat
    }
    if args.events:
        out_paths["Events DAT"] = args.events
    backup_output_files(out_paths)

    # Load system
    logger.info("Loading trajectory and topology.")
    u = mda.Universe(args.topology, args.trajectory)
    psel = u.select_atoms(args.selection1)  # original (may include H) for reporting/PDB write
    rsel = u.select_atoms(args.selection2)
    if len(psel) == 0 or len(rsel) == 0:
        logger.error("One or both selections are empty. s1='%s' (%d atoms), s2='%s' (%d atoms).",
                     args.selection1, len(psel), args.selection2, len(rsel))
        sys.exit(1)

    _log_system_specs(u, psel, rsel, args.begin, args.end, args.on_thresh, args.off_thresh,
                      args.events, args.min_dur, args.gap_fill, args.stride)

    # ---- Core calculation ----
    (atom_prob_p, atom_prob_r, avg_dist, contact_freq,
     res1, res2, p_residues, res_prob_p,
     bound_trace, times_ps, dmin_trace, dmean_trace) = calculate_proximity_and_contacts(
        u, args.selection1, args.selection2,
        args.begin, args.end, args.on_thresh, args.off_thresh, args.stride,
        trace_true_min=args.trace_true_min
    )

    # Heavy-only atoms for residue max-prob computation
    pheavy = u.select_atoms(f"{args.selection1} and not name H*")
    rheavy = u.select_atoms(f"{args.selection2} and not name H*")

    # Residue-level probabilities for BOTH groups = max(atom_prob per residue)
    resid_max_p = _residue_max_from_atom_probs(pheavy.atoms, atom_prob_p)
    resid_max_r = _residue_max_from_atom_probs(rheavy.atoms, atom_prob_r)

    # Move to last frame to write PDB snapshot
    if args.end is not None:
        u.trajectory[min(args.end - 1, len(u.trajectory) - 1)]
    else:
        u.trajectory[-1]

    # Ensure B-factor field exists; assign group-wise residue probabilities to ALL atoms in each group
    if not hasattr(u.atoms, 'tempfactors'):
        u.add_TopologyAttr('tempfactors')

    (psel + rsel).atoms.tempfactors = 0.0
    for a in psel.atoms:
        a.tempfactor = resid_max_p.get(a.resid, 0.0)
    for a in rsel.atoms:
        a.tempfactor = resid_max_r.get(a.resid, 0.0)

    # Write PDB (selection union) and contacts.dat
    _ = backup_file(args.output)
    (psel + rsel).write(args.output)

    _ = backup_file(args.datfile)
    write_residue_dat(resid_max_p, resid_max_r, psel, rsel, args.datfile)

    # Rough dt for windows/traces output (units neglected intentionally)
    dt_base = getattr(u.trajectory, "dt", None)
    dt_ps = args.dt_ps or _estimate_dt_ps(times_ps) or (dt_base * args.stride if dt_base is not None else None)
    logger.info("dt used for windows/traces: %s ps", f"{dt_ps:.6f}" if dt_ps is not None else "NA")

    # Plots
    if avg_dist is not None and contact_freq is not None:
        _ = backup_file(args.distmap)
        plot_distance_map(avg_dist, res1, res2, args.distmap, args.xlabel, args.ylabel, args.off_thresh, args.cmap_dist)
        _ = backup_file(args.contmap)
        plot_contact_probability_map(contact_freq, res1, res2, args.contmap, args.xlabel, args.ylabel, args.cmap_cont)
        logger.info("Distance map saved: %s", args.distmap)
        logger.info("Contact probability map saved: %s", args.contmap)
    else:
        logger.info("Contact maps skipped (insufficient residues/heavy atoms).")

    _ = backup_file(args.barplot)
    plot_residue_bar(p_residues, res_prob_p, args.barplot)
    logger.info("Residue-wise contact probability bar plot saved: %s", args.barplot)

    # Bound windows
    windows = _extract_windows(bound_trace, args.begin, dt_ps, args.min_dur, args.gap_fill, times_ps)
    if windows:
        logger.info("Bound windows (after hysteresis + debouncing):")
        logger.info(" start_f end_f start_ps end_ps duration_ps")
        for w in windows:
            logger.info(f"{w['start_f']:7d} {w['end_f']:6d} "
                        f"{('%.3f'%w['start_ps']) if w['start_ps'] is not None else ' NA':>9} "
                        f"{('%.3f'%w['end_ps']) if w['end_ps'] is not None else ' NA':>9} "
                        f"{('%.3f'%w['dur_ps']) if w['dur_ps'] is not None else ' NA':>12}")
    else:
        logger.info("No bound windows detected.")

    if args.events:
        _ = backup_file(args.events)
        _write_events_dat(windows, args.events)
        logger.info("Events DAT saved: %s", args.events)

    # Distance traces DAT
    if np.all(np.isfinite(times_ps)):
        t_out = times_ps
    elif dt_ps is not None:
        t_out = np.arange(len(dmin_trace), dtype=np.float64) * dt_ps
    else:
        t_out = np.full(len(dmin_trace), np.nan, dtype=np.float64)

    if args.distdat:
        _ = backup_file(args.distdat)
        _write_dist_traces(t_out, dmin_trace, dmean_trace, args.distdat)
        logger.info("Distance traces DAT saved: %s", args.distdat)

    # Final logs
    end_time = time.time()
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("End time: %s", time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(end_time)))
    elapsed = end_time - start_time
    logger.info("Total execution time: %.2f s (%.2f min)", elapsed, elapsed/60.0)
    logger.info("Output files:")
    logger.info(" - PDB: %s", args.output)
    logger.info(" - DAT: %s", args.datfile)
    if avg_dist is not None:
        logger.info(" - Distance map: %s", args.distmap)
        logger.info(" - Contact map: %s", args.contmap)
    logger.info(" - Bar plot: %s", args.barplot)
    if args.events:
        logger.info(" - Events DAT: %s", args.events)
    if args.distdat:
        logger.info(" - Distance traces DAT: %s", args.distdat)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
