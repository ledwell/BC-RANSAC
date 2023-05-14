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
1. Here is an example of the usage. _01_ is the directory number within the image directory that contains image pairs. _-d_ indicates that the directory number is next.
Directory name: `input/images/01/1.png`

`python imageAnalysis.py -d 01`
## Additional Notes
## Reference 