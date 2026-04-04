!/bin/bash

hicConvertFormat -m /home/hc0783.unt.ad.unt.edu/workspace/data/gm12878/GSE63525_GM12878_insitu_primary_replicate_combined_30.hic --inputFormat hic --outputFormat cool -o /home/hc0783.unt.ad.unt.edu/workspace/data/gm12878/gm12878.cool --resolutions 25000

hicConvertFormat -m /home/hc0783.unt.ad.unt.edu/workspace/data/gm12878/gm12878_25000.cool --inputFormat cool --outputFormat cool -o /home/hc0783.unt.ad.unt.edu/workspace/data/gm12878/gm12878_25000_KR.cool --correction_name KR
