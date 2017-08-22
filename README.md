
# Vehicle Detection and Tracking

The Project
---



```python
import os
import numpy as np
import glob
import cv2
import time
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import pickle
%matplotlib inline
```

## 1. Feature extraction

### 1.1 Color transformation


```python
def convert_color(img, color_space='BGR'):
        # Convert image to new color space (if specified)
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: 
        feature_image = np.copy(img)             

    # Return the feature vector
    return feature_image
```

### 1.2 Spatial binning


```python
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))                  
```

### 1.3 Color histogram


```python
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
```

### 1.4 HOG features


```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

### 1.5 Extract feature from one image


```python
# Define a function to extract features from a limage
def img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, spatial_size):
    file_features = []
    
    #Spatial feature extraction
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    
    #Color feature extraction
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    
    #HOG feature extraction
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    return file_features
```

### 1.6 Feature extraction for all images


```python
def extract_features(imgs, cspace='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file_p in imgs:
        
        file_features = []
        
        # Read in each imageone by one
        image = cv2.imread(file_p)
        
        # apply color conversion if other than 'BGR'
        feature_image = convert_color(image, color_space=cspace)
        
        #Get the features
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, spatial_size)
        
        #Append features to the list 
        features.append(np.concatenate(file_features))
        
        #Get the features and append features of flip image for augmentation
        feature_image=cv2.flip(feature_image,1) 
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, spatial_size)
        
        features.append(np.concatenate(file_features))
        
    return features # Return list of feature vectors
```

## 2.0 Classifier 

### 2.1 Data load


```python
images = glob.iglob('Features/**/*.png', recursive=True)

cars = []
notcars = []

for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)

print("Number of vehicle images: ", len(notcars))
print("Number of non-vehicle images: ", len(cars))
```

    Number of vehicle images:  8968
    Number of non-vehicle images:  8792



```python
#Data exploration
img = cv2.imread(image)
print("Shape", img.shape)
print("max", np.amax(img))
print("min", np.amin(img))
print("mean", np.mean(img))
```

    Shape (64, 64, 3)
    max 255
    min 7
    mean 100.578369141


### 2.2 Features extraction


```python
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size=(32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
```


```python
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

# Check the extraction time
t=time.time()

#Extract the car features
car_features = extract_features(cars, cspace=colorspace, 
                        spatial_size = spatial_size, hist_bins = hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

#Extract the non-cars features
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)    

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extraction...')
```

    502.51 Seconds to extraction...


### 2.3 Feature normalization


```python
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```


```python
with open('ScvScaler.pkl', 'wb') as fid:
    pickle.dump(X_scaler, fid)
```

### 2.4 Create train and validation data


```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                                    test_size=0.2, 
                                                    random_state=rand_state)

print("Train data shape: ", X_train.shape)
print("Validation data shape: ", X_test.shape)
```

### 2.5 Train the clasifier


```python
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()

svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
```


```python
with open('ScvClassifier.pkl', 'wb') as fid:
    pickle.dump(svc, fid)
```

## 3. Window search

### 3.1 Load test images and classifier


```python
images = glob.glob('test_images/straight_lines*.jpg')
images = images +  glob.glob('test_images/test*.jpg')

imgs = []

for idx, fname in enumerate(images):
    
    img = cv2.imread(fname)
    imgs.append(img)
```


```python
#Data exploration
print("Shape", img.shape)
print("max", np.amax(img))
print("min", np.amin(img))
print("mean", np.mean(img))
```

    Shape (738, 1280, 3)
    max 255
    min 0
    mean 107.832606072


### 3.2 Find cars in image


```python
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, 
              X_scaler, orient, pix_per_cell, cell_per_block, 
              spatial_size, hist_bins, color_space, hog_channel, 
              cells_per_step = 2,
              spatial_feat=True, hist_feat=True, hog_feat=True):
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
       
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
 
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
       
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = int((nxblocks - nblocks_per_window) // cells_per_step)
    nysteps = int((nyblocks - nblocks_per_window) // cells_per_step)
    
    #Create a list of sub-imgs and boundary boxes
    subimgs = []
    bboxs = []
    feat = []
    all_bboxs = []
        
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = int(yb*cells_per_step)
            xpos = int(xb*cells_per_step)
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))   
            test_prediction = svc.predict(test_features)
            
            #Create the boxes
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            bbox = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
            
            #Create all the boxes for debug
            all_bboxs.append(bbox)
            
            if test_prediction == 1:
                 
                bboxs.append(bbox)
                #for debug
                subimgs.append(subimg)
                feat.append(test_features)
                
    return bboxs, subimgs, feat, all_bboxs
```

### 3.3 Test


```python
ystart = 400
ystop = 656
save_subim = False

list_bboxs = []

for idx1, img in enumerate(imgs): 
        
    scale = 1.25
    ystart = 400
    ystop = 490
    cells_per_step = 1.0
    bboxs1, subimgs1, feat1, all_bboxs1 = find_cars(img, ystart, ystop, 
                    scale, svc, X_scaler, 
                    orient, pix_per_cell, 
                    cell_per_block, 
                    spatial_size, hist_bins, colorspace, hog_channel,
                    cells_per_step)
    
    scale = 2.0
    ystart = 390
    ystop = 590
    cells_per_step = 2.0
    bboxs2, subimgs2, feat2, all_bboxs2 = find_cars(img, ystart, ystop, 
                    scale, svc, X_scaler, 
                    orient, pix_per_cell, 
                    cell_per_block, 
                    spatial_size, hist_bins, colorspace, hog_channel,
                    cells_per_step)
    
    scale = 3.5
    ystart = 400
    ystop = 690
    cells_per_step = 1.0
    bboxs3, subimgs3, feat3, all_bboxs3 = find_cars(img, ystart, ystop, 
                    scale, svc, X_scaler, 
                    orient, pix_per_cell, 
                    cell_per_block, 
                    spatial_size, hist_bins, colorspace, hog_channel,
                    cells_per_step)
    
    
   

    subimgs = subimgs1 + subimgs2  + subimgs3 
    feats = feat1 + feat2 + feat3
    all_bboxs = [all_bboxs1, all_bboxs2, all_bboxs3]
    found_box = [bboxs1, bboxs2, bboxs3]
    
    list_bboxs.append(found_box)
    
    draw_img = np.copy(img)
    out_img_allb = np.copy(img)
    out_img = np.copy(img)
    
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    
    for colorIdsx, (bboxs, allbox) in enumerate(zip(found_box, all_bboxs)):
        
        color = colors[colorIdsx]
        
        for idx2, (bbox, subimg) in enumerate(zip(bboxs, subimgs)):
            out_img = cv2.rectangle(draw_img,bbox[0],bbox[1],color,6)
        
            if(save_subim):
                subimg = cv2.cvtColor(subimg, cv2.COLOR_YCrCb2BGR)
                cv2.imwrite("./output_images/image" + str(idx1) +" "+ str(idx2) + ".png", subimg)
        
        for bbox in allbox:
            out_img_allb = cv2.rectangle(out_img_allb,bbox[0],bbox[1],color,6)
        
        cv2.imwrite("./output_images/Search_image" + str(idx1) +".png", out_img_allb)
            
    #display origianl and transformation images
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,10))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=16)
    
    img = cv2.cvtColor(out_img_allb, cv2.COLOR_BGR2RGB)
    ax2.imshow(img)
    ax2.set_title('All boxes', fontsize=16)

    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    ax3.imshow(out_img)        
    ax3.set_title('find cars', fontsize=16)

```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)



![png](output_38_4.png)



![png](output_38_5.png)



![png](output_38_6.png)



![png](output_38_7.png)



```python
#Display the images with the searching boxes
f, axls = plt.subplots(1, 3, figsize=(15,15))

for idx, bboxs in enumerate(all_bboxs):
    
    draw_img = np.copy(imgs[4])
    out_img = np.copy(imgs[4])
    
    color = colors[idx]
    
    for bbox in bboxs:
        out_img = cv2.rectangle(draw_img,bbox[0],bbox[1],color,6)
    
    cv2.imwrite("./output_images/Search_image" + str(idx) +".png", out_img)
    
    axls[idx].set_title('Nr windows: {0}'.format(len(bboxs)), fontsize=16)
    axls[idx].imshow(out_img)
```


![png](output_39_0.png)


## 4. Multiple Detections & False Positives

### 4.1 Add heat and apply threshols 


```python
def add_heat(img, bbox_list):
    
    #place holder for heatmap
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
```


```python
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```

### 4.2 Draw boxes on labeled areas


```python
def draw_labeled_bboxes(img, labels):
        
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
    # Return the image
    return img
```

### 4.3 Test


```python

for idx1, (img, bboxs) in enumerate(zip(imgs, list_bboxs)): 
        
    # Add heat to each box in box list
    bbox = [item for sublist in bboxs for item in sublist]
    heat = add_heat(img,bbox)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    #display origianl and transformation images
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,10))
    ax1.imshow(heatmap, cmap='hot')
    ax1.set_title('Heat map', fontsize=16)
    ax2.imshow(labels[0])
    ax2.set_title('labels', fontsize=16)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    ax3.imshow(draw_img)
    ax3.set_title('Find cars', fontsize=16)
```


![png](output_47_0.png)



![png](output_47_1.png)



![png](output_47_2.png)



![png](output_47_3.png)



![png](output_47_4.png)



![png](output_47_5.png)



![png](output_47_6.png)



![png](output_47_7.png)


## 5. Pipeline


```python
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size=(32, 32)
hist_bins = 32
#Define all the scales to search for cars
scales = [1.25, 2.0, 3.5]
ystarts = [400, 390, 400]
ystops = [490, 590, 690]
cellsPerStepS = [1.0, 2.0, 1.0]

```


```python
with open('ScvClassifier.pkl', 'rb') as fid:
    svc = pickle.load(fid)
```


```python
with open('ScvScaler.pkl', 'rb') as fid:
    X_scaler = pickle.load(fid)
```


```python
def pipeline(img):
    
    bboxs = []
    
    #Find cars in the image with different scales
    for scale, ystart, ystop, cellsPerStep in zip(scales,ystarts,ystops,cellsPerStepS):
        
        #Find for cars
        bbox, subimg, feat, allbox = find_cars(img, ystart, ystop, 
                                        scale, svc, X_scaler, 
                                        orient, pix_per_cell, 
                                        cell_per_block, 
                                        spatial_size, hist_bins, 
                                        colorspace, hog_channel, cellsPerStep)
        #append the current boxes
        bboxs = bboxs + bbox
    
    # Add heat to each box in box list
    heat = add_heat(img,bboxs)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)
    
    print(heat.shape)
        
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
        
    return draw_img
```


```python
draw_img = pipeline(img)
draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
plt.imshow(draw_img)
```

    (738, 1280)





    <matplotlib.image.AxesImage at 0x7f8dd6fe8908>




![png](output_53_2.png)


### 6. Video Processing 


```python
clip_output = 'output_images/VehicleDetection.mp4'

clip = VideoFileClip("project_video.mp4")
clip = clip.fl_image(pipeline).subclip(25,45)
#clip = clip.fl_image(pipeline)
%time clip.write_videofile(clip_output, audio=False)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-520-7775d021b75b> in <module>()
          1 clip_output = 'output_images/VehicleDetection.mp4'
          2 
    ----> 3 clip = VideoFileClip("project_video.mp4")
          4 clip = clip.fl_image(pipeline).subclip(25,45)
          5 #clip = clip.fl_image(pipeline)


    ~/anaconda3/envs/VehicleDetection/lib/python3.5/site-packages/moviepy/video/io/VideoFileClip.py in __init__(self, filename, has_mask, audio, audio_buffersize, target_resolution, resize_algorithm, audio_fps, audio_nbytes, verbose, fps_source)
         79                                          target_resolution=target_resolution,
         80                                          resize_algo=resize_algorithm,
    ---> 81                                          fps_source=fps_source)
         82 
         83         # Make some of the reader's attributes accessible from the clip


    ~/anaconda3/envs/VehicleDetection/lib/python3.5/site-packages/moviepy/video/io/ffmpeg_reader.py in __init__(self, filename, print_infos, bufsize, pix_fmt, check_duration, target_resolution, resize_algo, fps_source)
         66 
         67         self.bufsize= bufsize
    ---> 68         self.initialize()
         69 
         70 


    ~/anaconda3/envs/VehicleDetection/lib/python3.5/site-packages/moviepy/video/io/ffmpeg_reader.py in initialize(self, starttime)
        101             popen_params["creationflags"] = 0x08000000
        102 
    --> 103         self.proc = sp.Popen(cmd, **popen_params)
        104 
        105 


    ~/anaconda3/envs/VehicleDetection/lib/python3.5/subprocess.py in __init__(self, args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env, universal_newlines, startupinfo, creationflags, restore_signals, start_new_session, pass_fds)
        674                                 c2pread, c2pwrite,
        675                                 errread, errwrite,
    --> 676                                 restore_signals, start_new_session)
        677         except:
        678             # Cleanup if the child failed starting.


    ~/anaconda3/envs/VehicleDetection/lib/python3.5/subprocess.py in _execute_child(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, env, startupinfo, creationflags, shell, p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite, restore_signals, start_new_session)
       1242                 errpipe_data = bytearray()
       1243                 while True:
    -> 1244                     part = os.read(errpipe_read, 50000)
       1245                     errpipe_data += part
       1246                     if not part or len(errpipe_data) > 50000:


    KeyboardInterrupt: 


### 6.1 Video Processing Plus


```python
from queue import *

class VehDetection:
    
    IMG_SHAPE = (720, 1280)
    
    def __init__(self, n, th):
        self.Heat = []
        self.n = n
        self.threshold = th
    
    def putHeatmap(self, heatmap):
        
        heatmap[heatmap > 0] = 1
        
        if(len(self.Heat) > self.n):
            self.Heat.pop(0)
            
        self.Heat.append(heatmap)
    
    def getHeatmap(self):
        
        heatmap = np.zeros(self.IMG_SHAPE).astype(np.float)
        
        for heat in self.Heat:
            heatmap = heatmap + heat
        
        heatmap = apply_threshold(heatmap,self.threshold)
                
        return heatmap        
```


```python
vehDetection = VehDetection(n=10, th=2)

def pipelinePlus(img):
    
    bboxs = []
    
    #Find cars in the image with different scales
    for scale, ystart, ystop, cellsPerStep in zip(scales,ystarts,ystops, cellsPerStepS):
        
        #Find for cars
        bbox, subimg, feat, allbox = find_cars(img, ystart, ystop, 
                                        scale, svc, X_scaler, 
                                        orient, pix_per_cell, 
                                        cell_per_block, 
                                        spatial_size, hist_bins, 
                                        colorspace, hog_channel,
                                        cellsPerStep)
        #append the current boxes
        bboxs = bboxs + bbox
       
    # Add heat to each box in box list
    heat = add_heat(img,bboxs)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)
    
    #Add and get the heat
    vehDetection.putHeatmap(heat)
    heatmap = vehDetection.getHeatmap()
       
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
        
    return draw_img
```


```python
#draw_img = pipelinePlus(img)
#draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
#plt.imshow(draw_img)
```


```python
clip_output = 'output_images/VehicleDetectionPlus3.mp4'

clip = VideoFileClip("project_video.mp4")
clip = clip.fl_image(pipelinePlus).subclip(20,24)
#clip = clip.fl_image(pipeline)
%time clip.write_videofile(clip_output, audio=False)
```

    [MoviePy] >>>> Building video output_images/VehicleDetectionPlus3.mp4
    [MoviePy] Writing video output_images/VehicleDetectionPlus3.mp4


     99%|█████████▉| 100/101 [01:06<00:00,  1.51it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output_images/VehicleDetectionPlus3.mp4 
    
    CPU times: user 1min 5s, sys: 280 ms, total: 1min 6s
    Wall time: 1min 8s



```python

```
