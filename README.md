# contact_analysis
```
usage: contact_analysis.py [-h] -s  -f  [-o] [-d] [-dm] [-cm] [-bp] [-dd] [-tm] [-ev] -s1  -s2  [-x] [-y] [-cmd] [-cmc] [-b] [-e] [-st] [-dt] [-on] [-off] [-md] [-gf] [-g]

Analyze contacts between two atom groups (heavy atoms only) with hysteresis; outputs PDB, DAT, maps, bar plot; optional event windows.

options:
  -h, --help            show this help message and exit

Inputs:
  -s , --topology       Topology file (str, required)
  -f , --trajectory     Trajectory file (str, required)

Outputs:
  -o , --output         Output PDB (str, default=contact.pdb)
  -d , --datfile        Residue probabilities DAT (str, default=contact.dat)
  -dm , --distmap       Distance map JPG (str, default=distance_map.jpg)
  -cm , --contmap       Contact map JPG (str, default=contact_map.jpg)
  -bp , --barplot       Residue bar plot JPG (str, default=contact_bar.jpg)
  -dd , --distdat       Distance traces DAT (str, default=dist_traces.dat)
  -tm, --trace_true_min
                        Distance traces use true minima via large-cap neighbor search (bool, default=False)
  -ev , --events        Bound windows output DAT (str, default=None=skip)

Selections:
  -s1 , --selection1    Group-1 selection (str, required)
  -s2 , --selection2    Group-2 selection (str, required)

Plotting:
  -x , --xlabel         X label (str, default=Residue index (group2))
  -y , --ylabel         Y label (str, default=Residue index (group1))
  -cmd , --cmap_dist    Colormap for distance map (str, default=magma)
  -cmc , --cmap_cont    Colormap for contact map (str, default=viridis)

Time frames:
  -b , --begin          Begin frame index (int, default=0)
  -e , --end            End frame index (int, default=None=last)
  -st , --stride        Use every k-th frame (int, default=1)
  -dt , --dt_ps         Time step in ps if missing (float, default=None)

Contact thresholds:
  -on , --on_thresh     ON threshold in Å (float, default=3.5)
  -off , --off_thresh   OFF threshold in Å (float, default=6.0)

Event windows:
  -md , --min_dur       Min bound duration in ps (float, default=50.0)
  -gf , --gap_fill      Merge gaps ≤ this (ps) (float, default=20.0)

Logging:
  -g , --log            Logfile (str, default=contact.log)
```
