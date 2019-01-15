# Tensorflow Object Detection API (1 of 2)

Learning about Tensorflow Object Detection API through "doing" instead of just "reading" someone else's code is perhaps the best way to learn.  This in OO is called an instance of an object.  If you can cover one model, you surely can do another and another.  The Lowest Common Denominator is perhaps a non-technical trait called "the love for programming" - seeded from Charles Petzold Windows Programming in 1990.

I discovered that Jupyter notebook has a nice feature of saving as Python Codes.  However, as we all know, displaying images off Jupyter notebook is so different from off the command line.  Also, I have never seen a webcam piece of code working off a Juypter notebook =) Some other learnings included resolving "cannot display plt off matplotlib backend " and of course some quick codes to polish up Python programming.  

Tensorflow 2.0 is coming in 2019, there will be more improvements and "what's in research folder" in object-detection could be merged as a base. The surprising find is that the detection speed is quite impressive.

A little knowledge is a dangerous thing.  You think you know but you actually don't. Then you know what you don't know and finally you find out enough to reduce the areas of unknowns.  

In the 80s, it's 3 Tier Client Server Computing, then comes WebServices.  Now it's BLOCKCHAIN and AI.  BLOCKCHAIN is still lightyears away but Deep Learning is already here to stay with commercial applications.  Do you need to know "Statistics" and "Machine Algorithms" to implement Deep learning ?  It's like saying do you need to know TCP socket programming for web services inorder to implement a web services project ?  The answer is of course no. Unless you are maintaining the codes for Web LOGIC web services.  Other developers just use the high level APs.  Tensorflow alone doesn't SOLVE commerical problems.  It's a combination of UNIX skills ( and surely no commercial app is running on MacOS or Windows 10 =) DEBUGGING skills and a logical sense of approach towards Systems Integration. 

Don't we love PIP3 install ? Without which, we end up paying for softwares just to try out.

Software:  UBUNTU 18.04, tensorflow 1.12, OpenCV 4.0.0-alpha, virtual environment, Python 3.6.7 [never ever go to 2.7]  
Hardware:  ASUS laptop, Intel® Core™ i7-7500U CPU @ 2.70GHz × 4, GeForce 940MX/PCIe/SSE2, 12 GB memory, Microsoft 2MB USB Webcam

##  Sample videos processed with the code

A crowded Orchard Road with lots of people detected [original video from youtube ]

<a href="http://www.youtube.com/watch?feature=player_embedded&v=uIKENd5VejM" target="_blank"><img src="https://img.youtube.com/vi/uIKENd5VejM/0.jpg" alt="webcam sample capture" width="240" height="180" border="10" /></a>

An car expressway in Singapore [original video from youtube ]

<a href="http://www.youtube.com/watch?feature=player_embedded&v=6qMIArxPo3k" target="_blank"><img src="https://img.youtube.com/vi/6qMIArxPo3k/0.jpg" alt="webcam sample capture" width="240" height="180" border="10" /></a>

##  Webcam video
<a href="http://www.youtube.com/watch?feature=player_embedded&v=8pmMGqQKLx0" target="_blank"><img src="https://img.youtube.com/vi/8pmMGqQKLx0/0.jpg" alt="webcam sample capture" width="240" height="180" border="10" /></a>

# Brief code walkthrough

## Preparation
1.	Download the ssd_mobilenet_v1_coco_11/frozen_inference_graph.pb from https://github.com/datitran/object_detector_app/tree/master/object_detection/ssd_mobilenet_v1_coco_11_06_2017 into object_detection/model directory

2.	Download the mscoco_label_map.pbtxt from https://github.com/datitran/object_detector_app/tree/master/object_detection/data
Into object_detection/data directory 

## Pseudocode

1.	Load labels
2.	Load Detection Graph
3.	Activate Web Camera 
4.	In a loop - Detect Objects - detect_objects(img, sess, detection_graph)
5.	Display modified image with Bounding Boxes

## Key Learnings

a.	Somehow the default plt backend is agg and with agg, plt cannot plot to the display. Hence the need for **plt.switch_backend('tkagg')**   
b.	Convert image via **cv2.cvtColor(img, cv2.COLOR_BGR2RGB)** before passing the image to detect.  Without this, the color is kind of BLUE-GREYISH  
c.	Somehow, **plt.pause(0.001)** seems to use less computer cycle than cv2.waitKey(1)

