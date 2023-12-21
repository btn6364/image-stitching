# Image Stitching using OpenCV. 

An implementation of the famous image stitching technique from scratch using OpenCV. 
Image stitching involves 4 crucial steps: 
1. Detect key points and descriptors. Then match them accordingly. 
2. Compute the homography matrix for image transformation. 
3. Warp image (Project one image to another image's plane using the homography matrix). 
4. Blend the warped images together. 

## Getting Started

### Dependencies
First, you need to install all the required dependencies using ```pip``` command. 
```
pip3 install -r requirements.txt
```

### Executing program
1. Save all your images in ```images``` folder. Also, the program only supports ```.jpg``` images.
2. To execute the program, run the following command. 
```
python3 image_stitching.py
```
3. After you ran the program, the stitched image will be saved in ```stitched.png```

## Authors

Contributors names and contact info:
* Bao Nguyen (btn6364@rit.edu)