----------------------------
What is this section about ?
---------------------------

Consider a Black Box where you are getting values from a proximity sensor (imagine anything) as a time-series data. You want to predict and do inference with those values for certain classes, using best approaches such just it is optimized to full possibility and doesn't create any overhead during real-time communications.

Assume you have a sensor, a microcontroller board (in this repository mostly everything is related to STM32-ULP).

Do something and infer/classify the raw-data, to know what's actually going on in the stream.

---------------------------

> A possible pipeline for the above tasks and constraints could be:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/99dabd7f-db1e-4a56-957b-356ae59fc311">


---------------------------

Some of the feature selection methods used here, in-general and in-specific to LinearSVC(model used here)

> In general approach :

1. mRMR - https://github.com/smazzanti/mrmr

2. SHAP(SHapley Additive exPlanations) explain the impact of different features on the output of the model. Its values show how much each 
feature contributes to the model's decision, providing insight into feature importance.

Features at the top have a higher impact on the model’s output(importance) compared to those at the bottom. The more the spread from 0, 
the larger the impact of that particular feature.

> Model specific approach :

SVM weighting - using the absolute coefficients. These weights/coefficients tells how important each feature is to the decision boundary 
created by the model. The larger the absolute value of the weight, the more significant the feature is for classification.

TO DO..

- Recursive Feature Elimination (RFE).

---------------------------
