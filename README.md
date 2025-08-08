# contact_analysis
```
usage: contact_analysis.py [-h] -f  -s  -s1  -s2  [-b] [-e] [-on] [-off] [-o] [-d] [-dm] [-cm] [-bp] [-x] [-y]

         This code analyzes the contact between two selected groups of atoms and outputs the following:
         - PDB file including both selections and contact probability as B-factor in selection-1
         - DAT file containing sorted contact probilites of each atom
         - JPG plots: Distance map, contact map, bar plot for residue-wise contact probabilities

options:
  -h, --help            show this help message and exit
  -f , --trajectory     Trajectory file (XTC, TRR, DCD, etc.)
  -s , --topology       Topology file (TPR, GRO, PDB, PSF, etc.)
  -s1 , --selection1    Group-1 (MDAnalysis string)
  -s2 , --selection2    Group-2 (MDAnalysis string)
  -b , --begin          Frame index to begin calculation (default: 0)
  -e , --end            Frame index to end calculation (default: last frame)
  -on , --on_thresh     On threshold in Angstroms for hysteresis scheme (default: 3.5 Å)
  -off , --off_thresh   Off threshold in Angstroms for hysteresis scheme (default: 6.0 Å)
  -o , --output         Output PDB file name (default: contact.pdb)
  -d , --datfile        Output DAT file name (default: contact.dat)
  -dm , --distmap       Output distance map file name (default: distance_map.jpg)
  -cm , --contmap       Output contact probability map file name (default: contact_map.jpg)
  -bp , --barplot       Contact probability bar plot file name (default: contact_bar.jpg)
  -x , --xlabel         X-axis label for distance/contact map (selection-2)
  -y , --ylabel         Y-axis label for distance/contact map (selection-1)

Copyright reserved by Saumyak Mukherjee
```
