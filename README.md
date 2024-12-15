# AAAI2025_iPCM
The official PyTorch implementation for 2025-AAAI-Integrating Personalized Spatio-Temporal Clustering for Next POI Recommendation.

If you have any questions, please feel free to issue or contact me by email. If you use our codes and datasets in your research, please cite.

# Overview
Location-Based Social Networks (LBSNs) offer a rich dataset of user activity at Points-of-Interest (POIs), making next POI recommendation a key task. Traditional algorithms face challenges due to broad searching scopes, affecting recommendation accuracy. Users tend to visit nearby POIs and show temporal concentration in their activities, reflecting personalized spatio-temporal clustering. However, individual user data may be insufficient to capture these clustering effects for personalized recommendations. In this paper, we propose an integrated Personalized Spatio-Temporal Clustering Model (iPCM) for next POI recommendation. The model learns this kind of personalized spatio-temporal clustering effect by using global historical trajectory data in conjunction with user feature embeddings. It integrates the features of personalized spatio-temporal clustering with the user’s trajectory, and completes the user’s POI recommendation through a Transformer encoding and MLP decoding. To enhance the accuracy of predictions, we add a module of probability adjustment. The experimental results on multiple datasets show that with the help of personalized spatio-temporal clustering, the proposed iPCM is superior to existing methods in various evaluation metrics.

![image](https://github.com/user-attachments/assets/1ae6f3a4-ea3c-4a08-8c27-4c7a7cc7f431)

# Running
（To cluster time, please run the time_process.py）
Run the run.py
