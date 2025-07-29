# Comparative Analysis of Machine Learning Models for Telco Customer Churn Prediction

---

## Introduction
This report evaluates the performance of eight machine learning models applied to the Telco Customer Churn dataset, a binary classification problem predicting whether a customer will churn ("Yes" or "No"). The models assessed include **Neural Network**, **K-Nearest Neighbors (KNN)**, **Naive Bayes**, **CatBoost**, **Random Forest**, **LightGBM**, **XGBoost**, **Support Vector Machine (SVM)**, and **Logistic Regression**. Performance is measured using **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Area Under the ROC Curve (AUC)**. Given the dataset's class imbalance (positive class weight ~2.76738), metrics like AUC, Recall, and F1 are critical for evaluating performance on the minority class ("Yes").

---

## Model Performance Summary
The table below summarizes the performance metrics for each model based on the test set results.

| Model             | Accuracy | Precision | Recall | F1 Score | AUC    |
| :---------------- | :------- | :-------- | :----- | :------- | :----- |
| Neural Network    | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000 |
| KNN               | 0.9808   | 0.9971    | 0.9303 | 0.9626   | 0.9975 |
| Naive Bayes       | 0.9623   | 0.9444    | 0.9115 | 0.9277   | 0.9799 |
| CatBoost          | 0.7846   | 0.5845    | 0.6488 | 0.6150   | 0.8388 |
| Random Forest     | 0.7868   | 0.5984    | 0.5952 | 0.5968   | 0.8290 |
| LightGBM          | 0.7385   | 0.5044    | 0.7694 | 0.6093   | 0.8362 |
| XGBoost           | 0.7363   | 0.5017    | 0.7775 | 0.6099   | 0.8373 |
| SVM               | 0.7200   | 0.4821    | 0.7587 | 0.5896   | 0.8357 |
| Logistic Regression | 0.7200   | 0.4821    | 0.7587 | 0.5896   | 0.8357 |

---

## Detailed Analysis

### 1. Neural Network
* **Performance:** Achieves perfect scores (Accuracy, Precision, Recall, F1, AUC = 1.0000). The confusion matrix shows no misclassifications (1034 true negatives, 373 true positives).
* **Analysis:** This performance is highly unusual and likely indicates **overfitting** or **data leakage** (e.g., identical training and test sets or errors in data splitting). Neural networks typically achieve AUC ~0.80–0.85 on this dataset.
* **Strengths:** Captures complex non-linear patterns.
* **Weaknesses:** Prone to overfitting, computationally intensive, requires careful tuning.

### 2. K-Nearest Neighbors (KNN)
* **Performance:** Excellent results with Accuracy = 0.9808, Precision = 0.9971, Recall = 0.9303, F1 = 0.9626, and AUC = 0.9975. Only 27 misclassifications (26 false negatives, 1 false positive).
* **Analysis:** KNN excels due to effective feature scaling and one-hot encoding. High Precision and Specificity (0.9990) indicate strong majority class performance, while Recall (0.9303) is robust for the minority class.
* **Strengths:** Simple, non-parametric, effective with proper preprocessing.
* **Weaknesses:** Sensitive to feature scaling, computationally expensive for large datasets.

### 3. Naive Bayes
* **Performance:** Strong performance with Accuracy = 0.9623, Precision = 0.9444, Recall = 0.9115, F1 = 0.9277, and AUC = 0.9799. 53 misclassifications (33 false negatives, 20 false positives).
* **Analysis:** Performs well despite assuming feature independence, which may not hold fully. High AUC and balanced metrics suggest robustness.
* **Strengths:** Fast, handles categorical features well, robust to class imbalance.
* **Weaknesses:** Limited by feature independence assumption.

### 4. CatBoost
* **Performance:** Accuracy = 0.7846, Precision = 0.5845, Recall = 0.6488, F1 = 0.6150, AUC = 0.8388. 303 misclassifications (131 false negatives, 172 false positives).
* **Analysis:** Offers a balanced trade-off between Precision and Recall. Competitive AUC indicates good discriminative ability.
* **Strengths:** Natively handles categorical features, robust to overfitting.
* **Weaknesses:** Lower Precision, indicating more false positives.

### 5. Random Forest
* **Performance:** Accuracy = 0.7868, Precision = 0.5984, Recall = 0.5952, F1 = 0.5968, AUC = 0.8290. 300 misclassifications (151 false negatives, 149 false positives).
* **Analysis:** Highest Accuracy among non-top models but lower Recall for the minority class. AUC is slightly lower than boosting models.
* **Strengths:** Robust, handles non-linear relationships, less sensitive to tuning.
* **Weaknesses:** Struggles with minority class, potentially due to class imbalance.

### 6. LightGBM
* **Performance:** Accuracy = 0.7385, Precision = 0.5044, Recall = 0.7694, F1 = 0.6093, AUC = 0.8362. 368 misclassifications (86 false negatives, 282 false positives).
* **Analysis:** High Recall but low Precision suggests many false positives. Competitive AUC.
* **Strengths:** Fast, scalable, effective for structured data.
* **Weaknesses:** Poor Precision for majority class.

### 7. XGBoost
* **Performance:** Accuracy = 0.7363, Precision = 0.5017, Recall = 0.7775, F1 = 0.6099, AUC = 0.8373. 371 misclassifications (83 false negatives, 288 false positives).
* **Analysis:** High Recall but low Precision, similar to LightGBM. Slightly better AUC than LightGBM.
* **Strengths:** Powerful for structured data, handles class imbalance with tuning.
* **Weaknesses:** Low Precision impacts F1 score.

### 8. Support Vector Machine (SVM)
* **Performance:** Accuracy = 0.7200, Precision = 0.4821, Recall = 0.7587, F1 = 0.5896, AUC = 0.8357. 394 misclassifications (90 false negatives, 304 false positives).
* **Analysis:** High Recall but very low Precision, indicating many false positives. AUC comparable to boosting models.
* **Strengths:** Effective for non-linear data with proper kernel.
* **Weaknesses:** Poor Precision, computationally intensive.

### 9. Logistic Regression
* **Performance:** Identical to SVM (Accuracy = 0.7200, Precision = 0.4821, Recall = 0.7587, F1 = 0.5896, AUC = 0.8357). 394 misclassifications (90 false negatives, 304 false positives).
* **Analysis:** Limited by linear assumptions, performs identically to SVM, suggesting possible configuration issues.
* **Strengths:** Interpretable, fast, good baseline.
* **Weaknesses:** Struggles with non-linear patterns, poor Precision.

---

## Ranking of Models (Best to Worst)
Models are ranked primarily by AUC, with F1 Score and Recall as tiebreakers due to class imbalance.

1.  **Neural Network (AUC = 1.0000):** Perfect performance, but likely due to overfitting or data leakage. Requires validation.
2.  **KNN (AUC = 0.9975):** Near-perfect AUC, high F1 (0.9626), and balanced metrics. Most reliable top performer.
3.  **Naive Bayes (AUC = 0.9799):** High AUC, strong F1 (0.9277), and balanced Precision/Recall. Efficient and robust.
4.  **CatBoost (AUC = 0.8388):** Competitive AUC, balanced F1 (0.6150), best among ensemble methods.
5.  **XGBoost (AUC = 0.8373):** High Recall (0.7775), slightly lower AUC than CatBoost, good for minority class.
6.  **LightGBM (AUC = 0.8362):** High Recall (0.7694), similar to XGBoost, slightly lower AUC.
7.  **Random Forest (AUC = 0.8290):** Lower AUC, struggles with minority class Recall (0.5952).
8.  **SVM (AUC = 0.8357):** High Recall but very low Precision (0.4821), tied with Logistic Regression.
9.  **Logistic Regression (AUC = 0.8357):** Identical to SVM, limited by linear assumptions, least effective.

---

## Summary and Recommendations

* **Best Model:** **KNN** (AUC = 0.9975, F1 = 0.9626) is the most reliable performer, offering near-perfect discriminative ability and balanced metrics. The **Neural Network's** perfect scores (AUC = 1.0000) are suspicious and likely indicate data leakage or overfitting; validation is critical.
* **Strong Alternatives:** **Naive Bayes** (AUC = 0.9799) is a fast and effective option. **CatBoost** (AUC = 0.8388) leads ensemble methods, followed by **XGBoost** (AUC = 0.8373) and **LightGBM** (AUC = 0.8362), which excel in Recall for the minority class.
* **Weaker Models:** **Random Forest**, **SVM**, and **Logistic Regression** underperform due to lower AUC and poor handling of the minority class (low Precision for SVM and Logistic Regression, low Recall for Random Forest).
* **Class Imbalance:** KNN and Naive Bayes balance Precision and Recall well, while XGBoost, LightGBM, and SVM prioritize Recall at the cost of Precision. Techniques like **SMOTE** could improve weaker models.

---

## Actionable Steps:

* **Validate Neural Network:** Check data splitting (`trainIndex`) and preprocessing to rule out leakage. Adjust size or decay to mitigate overfitting.
* **Optimize KNN:** Fine-tune `k` or distance metrics to enhance performance.
* **Tune Boosting Models:** Adjust hyperparameters (e.g., learning rate, tree depth) for CatBoost, XGBoost, and LightGBM to improve Precision.
* **Feature Selection:** Reduce the 50 features (post one-hot encoding) using feature importance to prevent overfitting.
* **SMOTE:** Apply `sampling = "smote"` in `trainControl` to boost minority class performance for Random Forest, SVM, and Logistic Regression.

---

## Conclusion
**KNN** is recommended as the best model for Telco Customer Churn prediction, pending validation of the Neural Network's unrealistic performance. Naive Bayes offers a lightweight alternative, while CatBoost, XGBoost, and LightGBM are strong ensemble options. Random Forest, SVM, and Logistic Regression are less effective but could improve with further tuning or preprocessing. Future work should focus on validating the Neural Network, optimizing top models, and reducing feature dimensionality to enhance performance and generalizability.
