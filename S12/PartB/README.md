## Details of JSON file:

### Images:

a. id : Unique id number for every image.

b. width : width of the image.

c. height : height of the image.

d. file_name : name of the file.

e. license : describe the image's license.

f. date_captured : Date at which image was taken.

### Annotations:

a. id : Annotation id.

b. image_id : Unique id of the image.

c. category_id : [0,num-categories]represents the category label. The value num-categories is reserved to represent the background category, if applicable.

d. Segmentation : (list[list(float)] or dictionary),

 A. list : represents a list of polygons, one for each connected component of the object, each list(float) is one simple polygon in the format of [x1, x2,....xn, yn]
 
 B. dict : represents the pre-pixel segmentation mask.
e. bbox : [x, y, width, height] of bounding box

f. is_crowd : 0 or 1, explains whether this instance is labeled as coco's crowd region.

g. area = area of the bounding box i.e (width*height)
