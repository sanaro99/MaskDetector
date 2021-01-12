# Mask Detector

This model uses Haar Cascade classifiers to detect Yes eyes, No nose, No mouth to ensure that mouth and nose are covered.

Traditional approach is to train a convolutional neural network on masked and non masked images.
Haar classifiers are faster as compared to a CNN an don't require model training time.

The below webcam stream recording shows how the model performs :-
\
\
![Webcam stream](webcam_record.gif)
