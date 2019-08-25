## Image Understanding with Deep Learning and Mathematics (IUDLM)
This is to understand images with deep learning approaches. As IUDLM tells, it involves Image Understanding
(object detection, localization, recognition, segmentation, understanding), Deep Learning (CNN, RNN, RL), and Mathematics (Optimization, Statistics). 
Our goal is to combine deep learning and object detection. For the overview of framework,
we refer to [Object Detection with Deep Learning: A Review -- Zhong-Qiu Zhao](https://arxiv.org/abs/1807.05511). We will borrow our machine learning algorithms and Cameo architecture.

#### Deep Learning
  1. Neural Network
  2. Probabilistic Graphical Model
  3. Solver

#### Image Understanding
  1. Object Detection
  2. Object Recognition
  3. Segmentation
  4. Localization

#### Mathematics:
  1. Optimization
  2. Statistics

#### The pipeline of theory -- Object Detection
  1. Region proposal based (R-CNN)
  2. Regression/Classification based (YOLO)
 
#### The pipeline of implementation (Object-Oriented)
  1. Prototype
  2. Optimization
  3. Established
  4. Standard

#### Object-Oriented
  1. Class
  2. Abstraction

#### Programming Language and Libraries
  1. Python/PyCharm
  2. Tensorflow
  3. OpenCV
  4. Numpy
  5. Pandas
  6. Matplotlib
  7. Sklearn
  8. Scipy
  9. MATLAB
  10. C++
  
class ClassName(object):
    
    def __init__(self):
        # variables

    def method(self):
        # operations
  
 
 ## Day 1
 Reference: [Selective Search for Object Recognition -- J.R.R. Uijlings](http://huppelen.nl/publications/selectiveSearchDraft.pdf)
  
 Problem: Generating possible object locations for use in object recognition
  
 Solution:  Selective Search
 
 ## Day 2
 Reference: [Efficient Graph-Based Image Segmentation -- Pedro F. Felzenszwalb](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)
 
 Problem: segmenting an image into regions
 
 Solution: Graph-Based Image Segmentation
 
 Reference: [Rich feature hierarchies for accurate object detection and semantic segmentation -- Ross Girshick](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
 
 Framework: R-CNN: Regions with CNN features
 
 Modules
  1. Region proposals
  2. Feature extraction
  3. Classification
  
Here we are focused on Region proposals. We have built the other modules. Once we can finish the region proposals module, we can build R-CNN, and its variants.

## Day 3
We refer to source code mentioned in Efficient Graph-Based Image Segmentation. We will write the prototpye using python.

## Day 4
We build a rough prototpye using Python.

## Day 5
We build the prototype using object-oriented programming.
Reference: Lifelong Machine Learning Systems: Beyond Learning Algorithms -- Daniel L. Silver

The goal is to sequentially retain learned knowledge and to selectively transfer that knowledge when learning a new task so as to develop more accurate hypotheses or policies.

## Day 6
We went over the Deep learning notes [cmu -- Deep learaning](http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Spring.2019/www/).
This is to introduce Deep learning with neural network.

[Rich feature hierarchies for accurate object detection and semantic segmentation -- Ross Girshick](http://www.rossgirshick.info/) proposes R-CNN -- Regions with CNN features.

## Day 7
Reference, Matching Networks for One Shot Learning -- Oriol Vinyals, consists of learning a class from a single labelled example.

## Day 8
We review Faster-RCNN and write down a summary. We focus on two Modules: Region Proposal Network (RPN) and Region of Interest (ROI) Pooling. We use keras to build a MiniVGGNet as the base network.

Reference, SCARLET-NAS: Bridging the gap Between Scalability and Fairness in Neural Architecture Search -- Xiangxiang Chu, proposes an Architeture search approach to bridge the gap between Scalability and Fairness with a linearly transformation. The problem can be converted into a multi-objective optimization problem. Mathematically, we can find the optimal architecture by sovling the optimization problem.

## Day 9
We extract the core idea from the review, [Recent Advances in Deep Learning for Object Detection -- Xiongwei Wu](https://arxiv.org/abs/1908.03673v1). Then we summarize the design guideline as a manual based on the reference and experience. We can build the computational model by constructing the blocks.

Deep Learning for Computer Vision
![](https://github.com/ZekiFayes/IUDLM/blob/master/DL4CV.png)

Layered Software Design
![](https://github.com/ZekiFayes/IUDLM/blob/master/System%20Design.png)

Object Detection Framework
![](https://github.com/ZekiFayes/IUDLM/blob/master/Object%20Detection.png)
