# Deep Learning Specialization

Welcome to the README of my submission for the Deep Learning Specialization! This certification, offered by DeepLearning.AI, is taught by top instructors Andrew Ng, Younes Bensouda Mourri, and Kian Katanforoosh on Coursera.

## Fair Usage Notice

Copying and pasting solutions directly from this repository for your own submissions on Coursera is against the honor code and terms of service of the platform. It is essential to uphold the principles of fairness, honesty, and integrity when undertaking online courses.

### Why is Fair Usage Important?

1. **Honor Code Violation:** Copying solutions without understanding or contributing to them is a direct violation of the honor code of Coursera and can lead to severe consequences.

2. **Learning Experience:** The primary purpose of these assignments is to enhance your learning experience. Copying solutions hinders your growth and understanding of the subject matter.

3. **Build Your Own Skills:** Attempting the assignments on your own helps you build crucial problem-solving and coding skills, which are essential in the real-world application of deep learning.

## Note on Ownership

The original notebooks belong to the institutions that offer these certifications under the MIT license. These notebooks are my submissions to the practical labs to complete these certifications.

## Your Responsibility

By using this repository, you acknowledge the importance of fair usage, agree to uphold the integrity of the learning process on Coursera, and recognize the ownership of the institutions over the content. Remember, the knowledge gained through these courses is more valuable than any certificate.

## Overview

- **Instructors:** Andrew Ng, Younes Bensouda Mourri, Kian Katanforoosh
  
- **Offered by:** DeepLearning.AI
  
- **Courses:**
  
    1. [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)
    2. [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization, and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)
    3. [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning)
    4. [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
    5. [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models?specialization=deep-learning)

- **Gained Skills:**
  
  1. Build and train deep neural networks, identify key architecture parameters, implement vectorized neural networks, and apply deep learning to various applications.
  2. Train test sets, analyze variance for deep learning applications, use standard techniques and optimization algorithms, and implement neural networks in TensorFlow.
  3. Build convolutional neural networks (CNNs) and apply them to detection and recognition tasks, utilize neural style transfer techniques to generate art, and apply algorithms to process image and video data.
  4. Build and train recurrent neural networks (RNNs), work with natural language processing (NLP) tasks and word embeddings, and utilize HuggingFace tokenizers and transformer models for named entity recognition (NER) and question answering.

## Certification Link

You can find my certification [here](https://coursera.org/share/b2c6848c8cc5df1fca38dc89a99ebbaf).

## Certification Notebooks and Labs

### Labs of Course 1: Neural Networks and Deep Learning

  #### Week 2: 
  - Logistic Regression with a Neural Network mindset:
    - You will build a logistic regression classifier from scratch to recognize cats. This assignment will step you through how to do this with a Neural Network mindset, and will also hone your intuitions about deep learning.
      - Deep_Learning_Specialization/Course_1/Week_2/Logistic_Regression_with_a_Neural_Network_mindset.ipynb

  #### Week 3:
  - Planar data classification with shallow neural network built from scratch:
      - Deep_Learning_Specialization/Course_1/Week_3/Planar_data_classification_with_one_hidden_layer.ipynb

  #### Week 4:
  - Building your Deep Neural Network: Step by Step
    - Implement Forward and backpropagation algorithms from scratch.
      - Deep_Learning_Specialization/Course_1/Week_4/1_Building_your_Deep_Neural_Network_Step_by_Step.ipynb
  - Build and train from scratch a deep L-layer neural network, and apply it to a cat/not-a-cat classifier
      - Deep_Learning_Specialization/Course_1/Week_4/2_Deep_Neural_Network_Application.ipynb

### Labs of Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

  #### Week 1: 
  - Initialization techniques:
    - Implement different weights initialization techniques used in deep learning including random, zeros, and He initialization, apply each one in a the deep L-layer neural network you built from scratch in Course 1 and analyze the effect of weights initialization to Speed up the convergence of gradient descent.
      - Deep_Learning_Specialization/Course_2/Week_1/1_Initialization_techniques.ipynb
  - Regularization:
    - Implement the L2 and Dropout regularization techniques from scratch and apply them in the deep L-layer neural network you built from scratch in Course 1 to reduce overfitting.
      - Deep_Learning_Specialization/Course_2/Week_1/2_Regularization.ipynb
  - Gradient Checking:
    - Implement gradient checking to verify the accuracy of your backpropagation implementation.
      - Deep_Learning_Specialization/Course_2/Week_1/3_Gradient_Checking.ipynb

  #### Week 2: Optimization methods
  - Implement from scratch the optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam and apply them to the deep L-layer neural network you built from scratch in Course 1, you will also implement from scratch the random mini-batch algorithm to accelerate convergence and improve optimization.
    - Deep_Learning_Specialization/Course_2/Week_2/Optimization_methods.ipynb

  #### Week 3: Tensorflow introduction
  - Explore Tensorflow, a deep learning framework that allows you to build neural networks more easily.
    - Deep_Learning_Specialization/Course_2/Week_3/Tensorflow_introduction.ipynb

### Labs of Course 4: Convolutional Neural Networks

  #### Week 1:
  - Convolution model Step by Step
    - Implement a convolutional neural network from scratch with its components (padding, stride, filter, ...).
      - Deep_Learning_Specialization/Course_4/Week_1/1_Convolution_model_Step_by_Step_v1.ipynb
  - Convolution model Application
    - Create a mood classifier using the TF Keras Sequential API. Also, build a ConvNet to identify sign language digits using the TF Keras Functional API.
      - Deep_Learning_Specialization/Course_4/Week_1/2_Convolution_model_Application.ipynb

  #### Week 2:
  - Residual Networks (ResNets)
    - Build a very deep convolutional network, using Residual Networks (ResNets) from scratch.
      - Deep_Learning_Specialization/Course_4/Week_2/1_Residual_Networks.ipynb
  - Transfer learning with MobileNet v1
    - Use transfer learning on a pre-trained MobileNetV2 to build an Alpaca/Not Alpaca classifier.
      - Deep_Learning_Specialization/Course_4/Week_2/2_Transfer_learning_with_MobileNet_v1.ipynb

  #### Week 3:
  - Car detection Autonomous driving application
    - Implement object detection using the very powerful YOLO model and explore low-level architecture of YOLO and postprocessing techniques of it for object detection and apply it to car detection dataset.
      - Deep_Learning_Specialization/Course_4/Week_3/1_Car_detection_Autonomous_driving_application.ipynb
  - Image segmentation Unet v2
    - Build your own U-Net, a type of CNN designed for quick, precise image segmentation, and use it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset.
      - Deep_Learning_Specialization/Course_4/Week_3/2_Image_segmentation_Unet_v2.ipynb

  #### Week 4:
  - Face Recognition
    - Build a face recognition system (one-shot learning, triplet loss, face encoding...).
      - Deep_Learning_Specialization/Course_4/Week_4/1_Face_Recognition.ipynb
  - Art Generation with Neural Style Transfer
    - Implement the neural style transfer algorithm and use it to generate novel artistic images (define style cost function and content cost function).
      - Deep_Learning_Specialization/Course_4/Week_4/2_Art_Generation_with_Neural_Style_Transfer.ipynb

### Labs of Course 5: Natural Language Processing and Sequence Models

  #### Week 1:

- Building a Recurrent Neural Network Step by Step
  - Implement basic RNNs, GRUs, and LSTMs networks from scratch using Numpy.
    - Deep_Learning_Specialization/Course_5/Week_1/1_Building_a_Recurrent_Neural_Network_Step_by_Step.ipynb

- Dinosaur Island Character-level Language Model
  - Build a Character level language model to generate names of dinosaurs.
    - Deep_Learning_Specialization/Course_5/Week_1/2_Dinosaur_Island_Character_level_language_model.ipynb

- Improvise a Jazz Solo with an LSTM Network
  - Build a model that uses an LSTM to generate your own jazz music using the flexible Keras Functional API.
    - Deep_Learning_Specialization/Course_5/Week_1/3_Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb

  #### Week 2:

- Operations on Word Vectors
  - Use pretrained word embeddings, explain how they capture relationships between words, measure similarity between word vectors using cosine similarity, and solve word analogy problems.
    - Deep_Learning_Specialization/Course_5/Week_2/1_Operations_on_word_vectors_v2a.ipynb

- Emojifier
  - Build a model that inputs a sentence and finds the most appropriate emoji to be used with this sentence.
    - Deep_Learning_Specialization/Course_5/Week_2/2_Emoji_v3a.ipynb

  #### Week 3:

- Neural Machine Translation with Attention
  - Build the RNN (Encoder-Decoder) architecture with attention as Neural Machine Translation (NMT) model to translate human-readable dates into machine-readable dates.
    - Deep_Learning_Specialization/Course_5/Week_3/1_Neural_machine_translation_with_attention_v4a.ipynb

- Trigger Word Detection
  - Construct a speech dataset and implement an algorithm for trigger word detection (wake word detection, for example "Hey Alexa turn On" for Amazon products).
    - Deep_Learning_Specialization/Course_5/Week_3/2_Trigger_word_detection_v2a.ipynb

  #### Week 4:

- Transformer Architecture
  - Build and train a Transformer model from scratch, the state-of-the-art deep learning architecture widely used in all revolutionizing Generative AI models Chatgpt, GPT4, Gemini, DALL-E, Sora...
    - Deep_Learning_Specialization/Course_5/Week_4/Transformer_Subclass_v1.ipynb

- Helpful Notebooks:
  - Preprocessing data for a transformer
  - Applications of the transformer in Named Entity Recognition and Question Answering tasks


## Build Your Career in AI and Data Science

If you are passionate about building a successful career in Artificial Intelligence and Data Science, I highly recommend enrolling via [this link](https://www.coursera.org/specializations/deep-learning). This certification, offered by renowned institutions and expert instructors, covers a wide range of topics and provides hands-on experience to sharpen your skills.

