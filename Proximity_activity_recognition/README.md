----------------------------
What is this section about ?
---------------------------

Consider a Black-Box where you are getting values from a proximity sensor, as a time-series data. You want to predict and do inference with those values for certain classes using best approaches, such just it is optimized to full possibility and doesn't create any overhead during real-time communications.

You have the sensor and a microcontroller board (in this repository mostly everything is related to STM32-ULP).

---------------------------

> A possible pipeline could be:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/d01518fb-2677-4baa-8f14-3a666e83e03d">


---------------------------

> A more representational pipeline could be:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/99dabd7f-db1e-4a56-957b-356ae59fc311">


---------------------------
Data Collection and Annotation
---------------------------

> One possible approach for data annotation after raw data collection from sensors:

__Label Studio - https://labelstud.io//__

<img width="600" alt="pipeline_annotated_TS_TCD" src="https://github.com/user-attachments/assets/0d8829f1-3bb6-4864-ac3d-3fb70903cbce">

---------------------------
__How the data could look like:__

<img width="600" alt="annotated_TS_TCD" src="https://github.com/user-attachments/assets/e7648c5d-e48f-4f69-a6f7-86090d56dac6">


---------------------------
Feature Engineering
---------------------------
<img width="600" src="https://github.com/user-attachments/assets/471ed7df-816f-4eca-a1d7-9850012cb563">

Ref: https://www.geeksforgeeks.org/what-is-feature-engineering/


Some of the feature selection methods used here, in-general and in-specific to LinearSVC(the model)


> In general approach :

1. minimum Redundancy - Maximum Relevance(mRMR) - https://github.com/smazzanti/mrmr

2. SHAP(SHapley Additive exPlanations) explain the impact of different features on the output of the model. Its values show how much each 
feature contributes to the model's decision, providing insight into feature importance.
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3e2300ad-b6dd-47e1-b41a-43cdceac1805">

Features at the top have a higher impact on the model’s output(importance) compared to those at the bottom. The more the spread from 0, 
the larger the impact of that particular feature.


> Model specific approach :

SVM weighting - using the absolute coefficients. These weights/coefficients tells how important each feature is to the decision boundary 
created by the model. The larger the absolute value of the weight, the more significant the feature is for classification.

TO DO..

- Recursive Feature Elimination (RFE).
- Improve model metrics of LinearSVC.
- Implement EfficientNet or EfficientNet-Lite
- Explore On-device learning
- Explore federated learning
---------------------------
