#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICU Patient Death Prediction System - Model Training Module
模型训练模块，负责机器学习模型的训练和评估

Copyright (c) 2023 Escoffier Zhou. All rights reserved.
This project is licensed under the MIT License. See LICENSE file for details.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("开始加载数据...")
    try:
        # 读取数据
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
        data = pd.read_csv(data_path)
        print(f"数据加载成功，形状: {data.shape}")
        
        # 定义分类特征
        categorical_cols = [
            'ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 
            'icu_type', 'apache_2_bodysystem', 'apache_3j_bodysystem'
        ]
        
        # 删除不需要的列
        drop_columns = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id', 'Unnamed: 83']
        data = data.drop(columns=drop_columns, errors='ignore')
        
        # 处理分类变量
        data[categorical_cols] = data[categorical_cols].fillna('Unknown')
        
        # 获取目标变量
        y = data['hospital_death'].values
        
        # 对分类变量进行独热编码
        data = pd.get_dummies(data, columns=categorical_cols)
        
        # 删除目标变量
        X = data.drop('hospital_death', axis=1)
        
        # 处理数值变量的缺失值
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        imputer = KNNImputer(n_neighbors=5)
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        # 标准化数值特征
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        print(f"特征处理完成，最终特征数量: {X.shape[1]}")
        print("特征列表:", list(X.columns))
        
        # 保存特征列表，用于预测时的特征对齐
        feature_list = list(X.columns)
        if not os.path.exists("model"):
            os.makedirs("model")
        with open(os.path.join("model", "feature_list.pkl"), 'wb') as f:
            pickle.dump(feature_list, f)
        
        # 保存数据处理器
        with open(os.path.join("model", "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join("model", "imputer.pkl"), 'wb') as f:
            pickle.dump(imputer, f)
            
        return X.values, y
        
    except Exception as e:
        print(f"数据处理过程中出错: {str(e)}")
        raise

def plot_roc_curve(model, X, y):
    """绘制ROC曲线"""
    # 预测概率
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 计算ROC曲线的点
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # 保存图像
    if not os.path.exists("model"):
        os.makedirs("model")
    plt.savefig(os.path.join("model", "roc_curve.png"))
    plt.close()
    
    print(f"\nROC曲线已保存，AUC值: {roc_auc:.4f}")

def train_and_save_model(X, y):
    """训练和保存模型"""
    try:
        print("\n开始��型训练...")
        
        # 初始化基础模型
        rf_clf = RandomForestClassifier(
            bootstrap=False, max_depth=None, max_features='sqrt', 
            min_samples_leaf=2, min_samples_split=10, n_estimators=200, 
            random_state=42, verbose=1
        )
        
        lgbm_clf = LGBMClassifier(
            colsample_bytree=1.0, lambda_l1=0.5, lambda_l2=0.1,
            learning_rate=0.05, max_depth=5, n_estimators=200,
            scale_pos_weight=1, subsample=0.8, random_state=42, 
            verbose=1
        )
        
        # 创建堆叠模型
        stacking_clf = StackingClassifier(
            estimators=[('rf', rf_clf), ('lgbm', lgbm_clf)],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            verbose=1
        )
        
        # 使用StratifiedKFold进行交叉验证
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 初始化评估指标列表
        accuracy_scores = []
        log_loss_scores = []
        auc_scores = []
        
        print("\n开始交叉验证...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            print(f"\n训练折次 {fold}/5")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # 训练模型
            stacking_clf.fit(X_train, y_train_fold)
            
            # 计算评估指标
            accuracy = stacking_clf.score(X_val, y_val_fold)
            y_pred_prob = stacking_clf.predict_proba(X_val)
            loss = log_loss(y_val_fold, y_pred_prob)
            
            # 计算AUC
            fpr, tpr, _ = roc_curve(y_val_fold, y_pred_prob[:, 1])
            fold_auc = auc(fpr, tpr)
            
            accuracy_scores.append(accuracy)
            log_loss_scores.append(loss)
            auc_scores.append(fold_auc)
            
            print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Log Loss: {loss:.4f}, AUC: {fold_auc:.4f}")
            
        # 输出整体评估结果
        print("\n交叉验证结果:")
        print(f"平均准确率: {np.mean(accuracy_scores):.4f} (±{np.std(accuracy_scores):.4f})")
        print(f"平均Log Loss: {np.mean(log_loss_scores):.4f} (±{np.std(log_loss_scores):.4f})")
        print(f"平均AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
        
        # 在全部数据上训练最终模型
        print("\n在全部数据上训练最终模型...")
        stacking_clf.fit(X, y)
        
        # 保存模型
        print("\n保存模型...")
        model_path = os.path.join("model", "stacking_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(stacking_clf, f, protocol=4)
        
        # 验证模型文件
        print("\n验证模型文件...")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        test_pred = loaded_model.predict_proba(X[:1])
        print(f"模型加载和预测测试成功，预测结果形状: {test_pred.shape}")
        
        # 输出分类报告
        y_pred = stacking_clf.predict(X)
        print("\n最终模型在全量数据上的分类报告:")
        print(classification_report(y, y_pred))
        
        # 绘制ROC曲线
        print("\n绘制ROC曲线...")
        plot_roc_curve(stacking_clf, X, y)
        
        print("\n模型训练和保存完成!")
        return stacking_clf
        
    except Exception as e:
        print(f"模型训练过程中出错: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 创建模型目录
        os.makedirs("model", exist_ok=True)
        
        # 加载和预处理数据
        X, y = load_and_preprocess_data()
        
        # 训练和保存模型
        model = train_and_save_model(X, y)
        
        print("\n所有步骤完成!")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
