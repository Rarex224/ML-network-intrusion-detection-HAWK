
# 🦅 HAWK: IoT Network Intrusion Detection

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

**HAWK** is a high-performance Network Intrusion Detection System (NIDS) specifically architected for IoT environments. By leveraging the **UNSW-NB15 dataset**, HAWK identifies and classifies cyber threats using optimized Machine Learning and Deep Learning pipelines.

---

## 💡 Inspiration & Origin
This project is inspired by the pioneering work of **Nour Moustafa** and the **UNSW-NB15** research team. The original codebase served as a foundation for exploring network anomaly detection.

### 🛠 What We Modified
While the core logic remains true to the original research, **HAWK** introduces two key technical refinements:
1.  **Feature Selection Logic:** We recalibrated the **Pearson Correlation thresholds**, narrowing down the feature set to the most statistically significant variables ($r > 0.3$) to reduce model noise and training time.
2.  **Model Hyperparameters:** We updated the classifier parameters (specifically for Random Forest and MLP) to better handle the imbalanced nature of the preprocessed dataset, resulting in higher accuracy for minority attack classes.

---

> [!IMPORTANT]
> **Original Repository:** [IoT-Network-Intrusion-Detection-System-UNSW-NB15](https://github.com/abhinav-bhardwaj/IoT-Network-Intrusion-Detection-System-UNSW-NB15)
> 
> **Research Citation:** N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems," 2015 MilCIS.



## 🚀 Quick Start
```bash
# Clone the repository
git clone [https://github.com/Rarex224/ML-network-intrusion-detection-HAWK.git](https://github.com/Rarex224/ML-network-intrusion-detection-HAWK.git)
cd ML-network-intrusion-detection-HAWK
pip install -r requirements.txt



Data Preprocessing
Cleaning: Handled missing values, resulting in 81,173 high-quality samples.

Encoding: Applied One-Hot-Encoding to proto, service, and state.

Scaling: Normalized features using MinMax Scaler.

📚 Citations
Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS. DOI: 10.1109/MilCIS.2015.7348942.
'@ | Out-File -FilePath README.md -Encoding utf8

2. Commit and Push the update
git add README.md
git commit -m "docs: add inspiration and modification details"
git push origin main


