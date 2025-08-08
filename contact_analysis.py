#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import argparse
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit, prange
import logging
import time
import os
import sys
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)
logging.getLogger('MDAnalysis.coordinates').setLevel(logging.WARNING)


def check_input_files(topology, trajectory):
    """
    Check if input files exist, exit with error if not found.
    """
    missing_files = []
    
    if not os.path.exists(topology):
        missing_files.append(f"Topology file: {topology}")
    
    if not os.path.exists(trajectory):
        missing_files.append(f"Trajectory file: {trajectory}")
    
    if missing_files:
        logger.error("ERROR: The following input files were not found:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error("Please check the file paths and try again.")
        sys.exit(1)
    
    logger.info("Input files found and validated.")


def backup_file(filepath):
    """
    Create a timestamped backup of a file if it exists.
    Backup format: filename_ddmmyyyy-hhmmss.ext
    """
    if os.path.exists(filepath):
        # Get file parts
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
        
        # Create backup filename
        backup_name = f"{name}_{timestamp}{ext}"
        if directory:
            backup_path = os.path.join(directory, backup_name)
        else:
            backup_path = backup_name
        
        # Copy file to backup
        shutil.copy2(filepath, backup_path)
        logger.info(f"Backed up existing file: {filepath} -> {backup_path}")
        
        return backup_path
    return None


def backup_output_files(output_files):
    """
    Backup all output files that exist.
    """
    logger.info("Checking for existing output files...")
    backed_up = []
    
    for file_desc, filepath in output_files.items():
        backup_path = backup_file(filepath)
        if backup_path:
            backed_up.append((file_desc, filepath, backup_path))
    
@njit(parallel=True)
def _min_dist_map(pos1, ridx1, pos2, ridx2, n1, n2):
    """
    Compute minimum distance map between two sets of heavy atoms grouped by residue.
    """
    m = np.full((n1, n2), np.inf, dtype=np.float32)
    # For each residue pair
    for i in prange(n1):
        for j in prange(n2):
            min_dist_sq = np.inf
            # Find minimum distance between atoms in residue i and j
            for k in range(len(pos1)):
                if ridx1[k] == i:
                    for l in range(len(pos2)):
                        if ridx2[l] == j:
                            d2 = np.sum((pos1[k] - pos2[l])**2)
                            if d2 < min_dist_sq:
                                min_dist_sq = d2
            m[i, j] = np.sqrt(min_dist_sq)
    return m


def calculate_proximity_and_contacts(u, selection1, selection2, begin=0, end=None, on_thresh=3.5, off_thresh=6.0):
    """
    Calculate per-atom proximity probabilities and residue-wise contact maps using a hysteresis scheme.
    """
    # Atom-wise selections for proximity calculation
    p = u.select_atoms(selection1)
    r = u.select_atoms(selection2)

    # Heavy atom selections for residue contact map
    g1 = u.select_atoms(f"{selection1} and not name H*")
    g2 = u.select_atoms(f"{selection2} and not name H*")
    res1 = list(g1.residues)
    res2 = list(g2.residues)

    logger.info(f"Selection 1 atoms: {len(p)} (heavy atoms: {len(g1)}, residues: {len(res1)})")
    logger.info(f"Selection 2 atoms: {len(r)} (heavy atoms: {len(g2)}, residues: {len(res2)})")

    # Residue-wise contact probability for selection1 residues
    p_residues = list(p.residues)
    n_residues = len(p_residues)
    resid_map_p = {res.resid: i for i, res in enumerate(p_residues)}
    ridx_p = np.array([resid_map_p[a.resid] for a in p.atoms], dtype=np.int32)
    res_contact_count = np.zeros(n_residues, dtype=np.int32)

    n1, n2 = len(res1), len(res2)

    # Check if contact map should be generated
    skip_contact_map = (n1 <= 1 or n2 <= 1)
    if skip_contact_map:
        avg_dist = None
        contact_freq = None
    else:
        resid_map1 = {r.resid: i for i, r in enumerate(res1)}
        resid_map2 = {r.resid: i for i, r in enumerate(res2)}
        ridx1 = np.array([resid_map1[a.resid] for a in g1.atoms], dtype=np.int32)
        ridx2 = np.array([resid_map2[a.resid] for a in g2.atoms], dtype=np.int32)

        # Initialize accumulators for residue map
        dsum = np.zeros((n1, n2), dtype=np.float32)
        ccnt = np.zeros((n1, n2), dtype=np.int32)
        cstate = np.zeros((n1, n2), dtype=np.bool_)

    n_atoms = len(p)
    contact_count = np.zeros(n_atoms, dtype=np.int32)
    cstate_atom = np.zeros(n_atoms, dtype=np.bool_)

    # Get trajectory slice info
    if end is None or end == -1:
        end = len(u.trajectory)
    total_frames = end - begin
    if total_frames <= 0:
        logger.error("No frames to process (begin >= end or empty trajectory).")
        sys.exit(1)

    frame_count = 0

    logger.info(f"Total trajectory frames: {len(u.trajectory)}")
    logger.info(f"Processing {total_frames} frames for proximity analysis...")

    # Process trajectory
    for ts in tqdm(u.trajectory[begin:end], total=total_frames):
        # Atom-wise proximity calculation
        p_pos = p.positions
        r_pos = r.positions

        box = u.dimensions if u.dimensions is not None else None
        dist_matrix = distance_array(p_pos, r_pos, box=box)

        min_distances = np.min(dist_matrix, axis=1)

        # Hysteresis contact update for atoms
        atom_on = min_distances < on_thresh
        atom_off = min_distances > off_thresh
        cstate_atom[atom_on] = True
        cstate_atom[atom_off] = False
        contact_count += cstate_atom.astype(np.int32)

        # Residue-wise contact update for selection1 residues
        resid_hits = np.bincount(ridx_p[cstate_atom], minlength=n_residues) > 0
        res_contact_count += resid_hits.astype(np.int32)

        # Residue-wise contact map calculation (only if not skipped)
        if not skip_contact_map:
            pos1 = g1.positions
            pos2 = g2.positions
            dmap = _min_dist_map(pos1, ridx1, pos2, ridx2, n1, n2)

            # Hysteresis contact update for residue pairs
            on = dmap < on_thresh
            off = dmap > off_thresh
            cstate[on] = True
            cstate[off] = False

            # Accumulate residue map data
            dsum += dmap
            ccnt += cstate.astype(np.int32)

        frame_count += 1

    # Calculate probabilities and averages
    atom_probabilities = contact_count.astype(np.float32) / frame_count
    if not skip_contact_map:
        avg_dist = dsum / frame_count
        contact_freq = ccnt / frame_count
    else:
        avg_dist = None
        contact_freq = None

    # Compute residue-wise contact probabilities
    residue_probabilities = res_contact_count.astype(np.float32) / frame_count
    return atom_probabilities, avg_dist, contact_freq, res1, res2, p_residues, residue_probabilities


def write_dat_file(atoms, probabilities, output_file):
    """Write proximity probabilities to DAT file."""
    with open(output_file, 'w') as f:
        f.write("# atomname_atomid resname_resid probability\n")
        for atom, prob in sorted(zip(atoms, probabilities), key=lambda x: x[1], reverse=True):
            f.write(f"{atom.name}{atom.id}\t{atom.resname}{atom.resid}\t{prob:.4f}\n")


def plot_residue_bar(residues, probabilities, output_file):
    """
    Plot residue-wise contact probability as a bar plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True, dpi=200)
    res_ids = [res.resid for res in residues]
    ax.bar(res_ids, probabilities, color='b', edgecolor='b', linewidth=0.5)
    ax.set_xlabel('Protein Residues', fontsize=22)
    ax.set_ylabel('Proximity probability', fontsize=22)
    ax.set_xlim(-4, res_ids[-1]+5)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=18)
    fig.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_map(avg_dist, res1, res2, output_file, xlabel, ylabel):
    """
    Plot distance map.
    """
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=200)
    
    # Distance map
    n1, n2 = avg_dist.shape
    im = ax.imshow(avg_dist, cmap='jet', aspect='auto', origin='lower', extent=[1, n2, 1, n1])
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=18)
    
    # Add colorbar for distance
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Distance (Å)', fontsize=22)
    cbar.ax.tick_params(labelsize=16)
    
    # Save figure
    fig.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_contact_probability_map(contact_freq, res1, res2, output_file, xlabel, ylabel):
    """
    Plot contact probability map.
    """
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=200)
    
    # Contact frequency map
    n1, n2 = contact_freq.shape
    im = ax.imshow(contact_freq, cmap='Reds', aspect='auto', origin='lower', vmin=0, vmax=1, extent=[1, n2, 1, n1])
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(labelsize=18)
    
    # Add colorbar for frequency
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Contact Probability', fontsize=22)
    cbar.ax.tick_params(labelsize=16)
    
    # Save figure
    fig.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="""
         This code analyzes the contact between two selected groups of atoms and outputs the following:
         - PDB file including both selections and contact probability as B-factor in selection-1
         - DAT file containing sorted contact probilites of each atom
         - JPG plots: Distance map, contact map, bar plot for residue-wise contact probabilities""",
        epilog="Copyright reserved by Saumyak Mukherjee",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-f'  , '--trajectory', metavar="", type=str  , required=True                   , help='Trajectory file (XTC, TRR, DCD, etc.)')
    parser.add_argument('-s'  , '--topology'  , metavar="", type=str  , required=True                   , help='Topology file (TPR, GRO, PDB, PSF, etc.)')
    parser.add_argument('-s1' , '--selection1', metavar="", type=str  , required=True                   , help='Group-1 (MDAnalysis string)')
    parser.add_argument('-s2' , '--selection2', metavar="", type=str  , required=True                   , help='Group-2 (MDAnalysis string)')
    parser.add_argument('-b'  , '--begin'     , metavar="", type=int  , default=0                       , help='Frame index to begin calculation (default: 0)')
    parser.add_argument('-e'  , '--end'       , metavar="", type=int  , default = None                  , help='Frame index to end calculation (default: last frame)')
    parser.add_argument('-on' , '--on_thresh' , metavar='', type=float, default=3.5                     , help='On threshold in Angstroms for hysteresis scheme (default: 3.5 Å)')
    parser.add_argument('-off', '--off_thresh', metavar='', type=float, default=6.0                     , help='Off threshold in Angstroms for hysteresis scheme (default: 6.0 Å)')
    parser.add_argument('-o'  , '--output'    , metavar="", type=str  , default='contact.pdb'           , help='Output PDB file name (default: contact.pdb)')
    parser.add_argument('-d'  , '--datfile'   , metavar="", type=str  , default='contact.dat'           , help='Output DAT file name (default: contact.dat)')
    parser.add_argument('-dm' , '--distmap'   , metavar="", type=str  , default='distance_map.jpg'      , help='Output distance map file name (default: distance_map.jpg)')
    parser.add_argument('-cm' , '--contmap'   , metavar="", type=str  , default='contact_map.jpg'       , help='Output contact probability map file name (default: contact_map.jpg)')
    parser.add_argument('-bp' , '--barplot'   , metavar="", type=str  , default='contact_bar.jpg'       , help='Contact probability bar plot file name (default: contact_bar.jpg)')
    parser.add_argument('-x'  , '--xlabel'    , metavar="", type=str  , default='Residue index (group2)', help='X-axis label for distance/contact map (selection-2)')
    parser.add_argument('-y'  , '--ylabel'    , metavar="", type=str  , default='Residue index (group1)', help='Y-axis label for distance/contact map (selection-1)')

    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("PROXIMITY AND CONTACT ANALYSIS STARTING")
    logger.info("=" * 80)
    logger.info(f"Start time: {time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(start_time))}")
    logger.info(f"Topology file: {args.topology}")
    logger.info(f"Trajectory file: {args.trajectory}")
    logger.info(f"Selection 1: {args.selection1}")
    logger.info(f"Selection 2: {args.selection2}")
    logger.info(f"Frame range: {args.begin} to {args.end if args.end is not None else 'end'}")
    logger.info(f"Contact on threshold: {args.on_thresh} Å")
    logger.info(f"Contact off threshold: {args.off_thresh} Å")

    # Check input files exist
    check_input_files(args.topology, args.trajectory)

    # Prepare output file list for backup
    output_files = {
        "PDB file": args.output,
        "DAT file": args.datfile,
        "Distance map": args.distmap,
        "Contact map": args.contmap,
        "Bar plot": args.barplot
    }
    
    # Backup existing output files
    backup_output_files(output_files)

    # Load universe
    logger.info("Loading trajectory and topology...")
    u = mda.Universe(args.topology, args.trajectory)

    # Get selection information
    p = u.select_atoms(args.selection1)
    r = u.select_atoms(args.selection2)

    # Calculate probabilities and contact maps
    probabilities, avg_dist, contact_freq, res1, res2, p_residues, residue_probabilities = calculate_proximity_and_contacts(
        u, args.selection1, args.selection2,
        args.begin, args.end, args.on_thresh, args.off_thresh
    )

    logger.info("Writing PDB with contact probabilities in B-factor column...")
    both_selections = p + r 

    # Set frame for output - use last processed frame
    if args.end is not None:
        u.trajectory[min(args.end-1, len(u.trajectory)-1)]
    else:
        u.trajectory[-1]

    # Add tempfactors attribute if not present
    if not hasattr(u.atoms, 'tempfactors'):
        u.add_TopologyAttr('tempfactors')

    # Set tempfactors: probabilities for p, zero for r
    both_selections.atoms.tempfactors = 0.0
    p.atoms.tempfactors = probabilities

    # Write output files
    both_selections.write(args.output)
    write_dat_file(p.atoms, probabilities, args.datfile)

    # Plot contact maps (only if both selections have >1 residue)
    if avg_dist is not None and contact_freq is not None:
        plot_distance_map(avg_dist, res1, res2, args.distmap,
                         args.xlabel, args.ylabel)
        plot_contact_probability_map(contact_freq, res1, res2, args.contmap,
                                   args.xlabel, args.ylabel)
        logger.info(f"Distance map saved: {args.distmap}")
        logger.info(f"Contact probability map saved: {args.contmap}")
    else:
        logger.info("Contact maps skipped (one or both selections have ≤1 residue)")

    # Plot residue-wise contact probability bar plot
    plot_residue_bar(p_residues, residue_probabilities, args.barplot)
    logger.info(f"Residue-wise contact probability bar plot saved: {args.barplot}")

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"End time: {time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(end_time))}")
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    logger.info("Output files:")
    logger.info(f"  - PDB: {args.output}")
    logger.info(f"  - DAT: {args.datfile}")
    if avg_dist is not None:
        logger.info(f"  - Distance map: {args.distmap}")
        logger.info(f"  - Contact map: {args.contmap}")
    logger.info(f"  - Bar plot: {args.barplot}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
