# Lung tumor detection

THe project is devoted to the lungs tumor detection on computer tomography images using custom deep learning algorithm.



## Data preparation 

Composed by the [LUng Nodule Analysis (LUNA) 2016](https://luna16.grand-challenge.org/Data/) competition commiettee, the original projet dataset contains several files with distinct information, that should be augmented. The file `candidates.csv` contains spatial coordinates of all suspicious nodule candidates' centers, but lack sizes. Luckily, these dimensions could be retrieved from another file, `annotations.csv`, containig all tumor-class nodules, which were manually verified by certified CT-analysts.
