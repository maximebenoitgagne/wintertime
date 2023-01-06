Welcome! This repository contains the model and the scripts used to generate the figures in
Benoît-Gagné et al.,
"The overlooked importance of wintertime survival of phytoplankton".

Directory structure:

- The model itself is contained in the following directories: doc, eesupp, jobs, lsopt, model, optim, pkg, tools, utils, and verification.
- The configuration of the model is in the directory gud_1d_35+16.
- The jupyter notebook wintertime.ipynb generates the figures in the directory figures_progress.
- The jupyter notebook wintertime_supmat.ipynb generates  the figures in the directory figures_supmat_progress.
- Data for the generation of the figures (Data S1 to S8) including model output are openly accessible in the directory data.

Model:

- The model is a one-dimensional configuration of the biogeochemical/ecosystem model of Dutkiewicz et al. (2021) in *Glob. change biol.* The tracers are mixed by the MIT general circulation model (MITgcm, Marshall et al., 1997 in *JGR*). The paper Benoît-Gagné et al. describes some modifications relative to Dutkiewicz et al. (2021).

Datasets:

- data/DataS1_observations_IBCAO_1min_bathy.mat:
Bathymetry from International BAthymetric Chart of the Arctic Ocean (IBCAO) Version 3.0, Jakobsson et al. (2012). https://doi.org/10.1029/2012GL052219, details in MetadataS1.pdf.
- data/DataS2_observations_Qikiqtarjuaq:
*In situ* observations from the Qikiqtarjuaq sea ice camps 2015 and 2016 (67.4797°N,-63.7895°E). It contains a subset of the files available in the dataset Massicotte et al. (2019). https://doi.org/10.17882/59892. The paper related to this dataset is Massicotte et al. (2020). https://doi.org/10.5194/essd-12-151-2020, details in MetadataS2.pdf.
- data/DataS3_observations_CCGS_Amundsen:
*In situ* data sampled onboard the CCGS Amundsen, details in MetadataS3.pdf.
- data/DataS4_output_nemo_lim3:
Model output generated for this study using NEMO 3.6 (Madec et al., 2017,  https://doi.org/10.5281/zenodo.3248739) coupled with LIM 3.6 (Rousset et al., 2015, https://doi.org/10.5194/gmd-8-2991-2015), details in MetadataS4.pdf.
- data/DataS5_observations_davis_strait.csv:
*In situ* observations from 5 data points in Davis Strait, Colombo et al. (2020). [https:doi.org/10.1016/j.gca.2020.03.012](https:doi.org/10.1016/j.gca.2020.03.012), details in MetadataS5.pdf.
- data/DataS6_output_cgrf:
Model output generated using CGRF (Smith et al., 2014, [https:doi.org/10.1002/qj.2194](https:doi.org/10.1002/qj.2194)). For this study, data was selected for the location of the Qikiqtarjuaq sea ice camp 2016 (67.4797°N,-63.7895°E) and for the year 2016, details in MetadataS6.pdf.
- data/DataS7_literature.csv:
Observations from literature (Lacour et al., 2017. [https:doi.org/10.1002/lno.10369](https:doi.org/10.1002/lno.10369)), details in MetadataS7.pdf.
- data/DataS8_output_mitgcm:
Simulated data generated for this study, details in MetadataS8.pdf.

Notes:

Some data files are larger than 100 GB.
Hence, if Git LFS is not installed, the files larger than 100 GB will be replaced with placeholders after cloning the project.

The exact procedure I used to deploy the code on a supercomputer of the Digital Research Alliance of Canada with the SLURM workload manager is

```
module load git-lfs
git_lfs clone git@github.com:maximebenoitgagne/gud_groups.git
```

The procedure to run the model is in the README of the directory gud_1d_35+16.

I thank Achim Randelhoff for their GitHub project [ice-edge](https://github.com/poplarShift/ice-edge) and Laure Vilgrain for their GitHub project [copepods-true-colors](https://github.com/laurvi/copepods-true-colors/tree/v1.2) which have often served me as an example.

Let me know if you have any requests or comments. You can contact me via [ResearchGate](https://www.researchgate.net/profile/Maxime-Benoit-Gagne).
