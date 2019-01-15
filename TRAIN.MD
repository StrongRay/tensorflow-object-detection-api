# Training new object

Train a customised object for Object Detection

Reference: 
1. https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

## Folder Structure

Under home/object-detection/object_detection

![image](https://github.com/StrongRay/tensorflow-object-detection-api/blob/master/A2.png)


# A.   Prepare the data

## 1)     Extract images from Video

A video will definitely help provide enough angles of an object.  My idea is to hang an object to be detected on a string against a white background, then rotate the object and use webcam to capture the video.  Extract the frames from the video.  If this is done slowly, any object can be captured via such a method, perhaps on a rotating pedestral.

1. Install utility to capture video of object [640x480] -  `guvcview` 
2. Convert video into jpegs  - `ffmpeg -i video.mkv whale%03d.jpg` 
   264 images were extracted

## 2)    Annotate each image

Annotation is nothing but tediously defining a boundary border covering the object of interest and then saving with a class name.    Before doing this, I manually split the dataset into 2 sets ( 80% (211 jpegs) for training & 20% (53 jpegs) for evaluation ) by moving filenames by lots – eg: WHALE002.jpg WHALE012.jpg etc .. In future, can write a tool to randomly extract 20% and renaming each file.  A tool is used for this task.

### a.  Prepare the utility to do annotation

```git clone https://github.com/tzutalin/labbelImg.git
sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements linux-python3.txt
make qt5py3
```
### b.  Annotate the images

In labelImg directory, execute python3 labelImg.py and navigate to the eval directory and process each image by creating a rectbox and saving it

![image](https://github.com/StrongRay/tensorflow-object-detection-api/blob/master/A1.png)


