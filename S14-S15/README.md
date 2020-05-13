# Monocular Depth Estimation and Segmentation

## Group Members:

1. Abhijit Mali
2. Gunjan Deotale
3. Sanket Maheshwari
4. Sanjeev Raichur

## Dataset

Link: https://drive.google.com/drive/folders/1MST5DUffe3h9Q4B-x7tpNxXl4q4_E8ah

## Dataset Statistics:

### Kinds of images (fg, bg, fg_bg, masks, depth)

fg :- Different Man, Woman, kids, group of person(for background transparency we have taken png images) bg :- We restricted background to library images(for restricting size of image we have taken jpg images) fg_bg :- bg superposed over fg (for restricting size of images we have taken jpg images) masks :- masks extracted from fg images(we have taken grayscale images)(.jpg) depth :- We have extracted depth images from fg_bg using nyu model(for restricing size of images we have taken grayscale images extracted from colormap)(.jpg)

### Total images of each kind
fg :- 200(flip + no flip) bg :- 100 fg_bg :- 392783 masks :- 392469 depth :- 394673

##### fg_bg : 160x160x3
##### fg :  80 max size and min size is resized based on aspect ratio
##### mask: 160x160
##### depth images: 160x160


### The total size of the dataset :-
9182546 Output/ 6446124 Output/OverlayedImages/ 1119541 Output/OverlayedMasks/ 1616796 Output/DepthImage/

### Mean/STD values for your fg_bg, masks and depth images
#### fg_bg :- (BGR format) 
- Mean: - [0.3234962448835791, 0.3776562499540454, 0.4548452917585805]
- SD: - [0.22465676724491895, 0.2299902629415973, 0.23860387182601098]

#### masks :- (BGR format) 
- Mean: - [0.07863663756127236, 0.07863663756127236, 0.07863663756127236]
- SD: - [0.2541994994472449, 0.2541994994472449, 0.2541994994472449]

#### depth :- (BGR format) 
- Mean: - [0.2943823440611593, 0.2943823440611593, 0.2943823440611593] 
- SD: - [0.15619204938398595, 0.15619204938398595, 0.15619204938398595]

### Show your dataset the way I have shown above in this readme

##### Background Images 
![bg](https://github.com/sanjeev29/EVA-4/blob/master/S14-S15/sample/bg.png)

##### Foreground Images 
![fg](https://github.com/sanjeev29/EVA-4/blob/master/S14-S15/sample/fg.png)

##### Foreground Masks 
![fg_mask](https://github.com/sanjeev29/EVA-4/blob/master/S14-S15/sample/fg_mask.png)

##### Foreground+Background 
![fg_bg](https://github.com/sanjeev29/EVA-4/blob/master/S14-S15/sample/fg_bg.png)

##### Foreground+Background Mask 
![fg_bg_mask](https://github.com/sanjeev29/EVA-4/blob/master/S14-S15/sample/fg_bg_mask.png)

##### DepthMap 
![fg_bg_depth](https://github.com/sanjeev29/EVA-4/blob/master/S14-S15/sample/fg_bg_depth.png)

### Dataset creation description

#### How were fg images created with transparency?
Used remove background feature from Microsoft PowerPoint.

#### How were masks created for fg images?
Mask images are nothing but alpha channels of images. So we extracted masks using OpenCV as follows

```
image = cv2.imread("Foregroundimg.png", cv2.IMREAD_UNCHANGED) imagealpha = image[:,:,3] cv2.imwrite("ForegroundMask.jpg", imagealpha)
```

#### How did you overlay the fg over bg and created 20 variants?
1. First all background images were resized to 160x160
2. All foreground images were resized to 80(left and right) and top and bottom was reshaped as per aspect ratio
3. Images were randomly placed by choosing starting x,y randomly on background, but also making sure that foreground image does not go out of background image

#### How did you create your depth images?
Referred the following link for generating depth images. However neccessary changes were done to save images after depth images were generated.
Link: https://github.com/ialhashim/DenseDepth

#### How full dataset was created?
1. Downloaded 100 images of library and saved it as background images(bg)
2. Downloaded 100 images of humans and removed background and saved it as foreground images(fg)
3. The bg and fg images were resized
4. For each of the fg image, we created a mask(fg_mask)
5. We created an overlay of fg and bg images by randomly placing fg images(normal and flipped) on bg image
6. Mask was created for these overlay images
7. Depth images were generated for the overlay images
