# **Homography Estimation - Comparison of RANSAC and its Variants**
## About this Project
Structure from Motion (SFM) is the process of estimating a 3D model from a set of 2D images. SFM can be used in applications such as 3D scannning and visualization. We focused our reseached on SFM in UAV application. However, due to the scope of our project being to large compared to the time we had, we decided to consider Homography, which can be considered a part of the SFM pipeline. Homography relates the transformation between two views in the same planer surface. These steps are used in the first few steps of the SFM pipeline.
## Operating System 
- Windows
- Linux
- macOS
## Getting Started
### Prerequisites
- Python 3.7.0 or later
- OpenCV
- NumPY
- Matplotlib
- SciPy
- glob
### Installation
#### Windows
Download the zip file from [GitHub](https://github.com/ledwell/seniorProject).

Unzip the zip file to your project directory.
#### Linux
Git clone the repository.

`git clone https://github.com/ledwell/seniorProject.git`
#### macOS
Git clone the repository. 

`git clone https://github.com/ledwell/seniorProject.git`
## Usage
1. Here is an example of the usage. _01_ is the directory number within the image directory that contains image pairs. _-d_ indicates that the directory number is next. Below is an example of the output.
Directory name: `input/images/01/1.png`

`python imageAnalysis.py -d 01`

![Example of RANSAC in Homography](https://github.com/ledwell/seniorProject/blob/main/output/images/02/RANSAC_inlier_matches.png)
## Additional Notes
## Reference 
[1] https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
[2] https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
[3] http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
[4] https://stackoverflow.com/questions/28717054/calculating-sharpness-of-an-image
[5] https://stackoverflow.com/questions/24671901/does-there-exist-a-way-to-directly-figure-out-the-smoothness-of-a-digital-imag
[6] https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
[7] https://github.com/MarshalLeeeeee/Tamura-In-Python/blob/master/tamura-numpy.py