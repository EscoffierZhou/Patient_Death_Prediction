#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICU Patient Death Prediction System - Incremental Learning Module
增量学习模块，负责模型的在线更新和性能监控

Copyright (c) 2023 Escoffier Zhou. All rights reserved.
This project is licensed under the MIT License. See LICENSE file for details.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle
import os
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class IncrementalLearning:
    def __init__(self):
        """初始化增量学习模型"""
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        self.log_file = os.path.join(self.model_dir, "training_log.json")
        
        # 加载训练日志
        self.training_log = self.load_training_log()
        
        # 获取上次训练的最大ID
        self.last_trained_id = self.training_log.get('last_trained_id', 0)
        print(f"上次训练的最大ID: {self.last_trained_id}")
        
        # 尝试加载已有模型
        model_path = os.path.join(self.model_dir, "incremental_model.pkl")
        if os.path.exists(model_path):
            print("加载已有模型...")
            with open(model_path, 'rb') as f:
                saved_state = pickle.load(f)
                self.model = saved_state['model']
                self.scaler = saved_state['scaler']
                self.feature_names = saved_state['feature_names']
                self.imputer = saved_state.get('imputer')  # 添加imputer
        else:
            print("创建新的增量学习模型...")
            self.model = SGDClassifier(loss='log_loss', 
                                     learning_rate='adaptive',
                                     eta0=0.01,
                                     random_state=42)
            self.scaler = StandardScaler()
            self.imputer = KNNImputer(n_neighbors=5)
            self.feature_names = None
    
    def load_training_log(self):
        """加载训练日志"""
        try:
            # 确保model目录存在
            os.makedirs(self.model_dir, exist_ok=True)
            
            if os.path.exists(self.log_file):
                # 检查文件是否为空
                if os.path.getsize(self.log_file) == 0:
                    initial_log = {'last_trained_id': 0, 'training_history': []}
                    with open(self.log_file, 'w', encoding='utf-8') as f:
                        json.dump(initial_log, f, indent=4)
                    return initial_log
                    
                try:
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    print("训练日志文件损坏，创建新的日志文件")
                    initial_log = {'last_trained_id': 0, 'training_history': []}
                    with open(self.log_file, 'w', encoding='utf-8') as f:
                        json.dump(initial_log, f, indent=4)
                    return initial_log
            else:
                # 如果文件不存在，创建新的日志文件
                initial_log = {'last_trained_id': 0, 'training_history': []}
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_log, f, indent=4)
                return initial_log
            
        except Exception as e:
            print(f"加载训练日志时出错: {str(e)}")
            return {'last_trained_id': 0, 'training_history': []}
    
    def save_training_log(self, new_max_id, metrics, num_samples):
        """保存训练日志"""
        self.training_log['last_trained_id'] = new_max_id
        
        training_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'samples_trained': num_samples,
            'id_range': f"{self.last_trained_id + 1}-{new_max_id}",
            'metrics': metrics
        }
        self.training_log['training_history'].append(training_record)
        
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=4)
    
    def process_data(self, data=None, sample_size=None):
        """处理数据，如果没有提供数据，则读取CSV文件"""
        try:
            if data is None:
                # 读取数据
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
                raw_data = pd.read_csv(csv_path)
                
                # 只选择ID大于上次训练最大ID的数据
                new_data = raw_data[raw_data['patient_id'] > self.last_trained_id].copy()
                
                if len(new_data) == 0:
                    print("没有新的数据可供训练")
                    return None, None, None
                
                print(f"发现 {len(new_data)} 条新数据")
                
                # 如果指定了样本数量，随机选择指定数量的样本
                if sample_size and sample_size < len(new_data):
                    new_data = new_data.sample(n=sample_size, random_state=42)
                    print(f"随机选择 {sample_size} 条数据进行训练")
                
                # 记录最大ID
                max_id = new_data['patient_id'].max()
                
                # 删除不需要的列
                new_data.drop(['encounter_id', 'hospital_id', 'Unnamed: 83'], axis=1, inplace=True)
                data = new_data.copy()
            else:
                max_id = None
                data = pd.DataFrame([data]) if isinstance(data, dict) else data
            
            # 保存patient_id列
            patient_ids = data['patient_id'] if 'patient_id' in data.columns else None
            if 'patient_id' in data.columns:
                data = data.drop('patient_id', axis=1)
            
            # 定义分类特征
            categorical_cols = [
                'ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 
                'icu_type', 'apache_2_bodysystem', 'apache_3j_bodysystem'
            ]
            
            # 处理分类变量的缺失值
            data[categorical_cols] = data[categorical_cols].fillna('Unknown')
            
            # 对分类变量进行独热编码
            data = pd.get_dummies(data, columns=categorical_cols)
            
            # 获取数值列（不包括独热编码列和目标变量）
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'hospital_death']
            
            # 使用imputer填充缺失值
            if not hasattr(self.imputer, 'statistics_'):
                data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
            else:
                data[numeric_cols] = self.imputer.transform(data[numeric_cols])
            
            # 第一次运行时保存特征名称
            if self.feature_names is None:
                self.feature_names = data.drop(['hospital_death'], axis=1).columns
            else:
                # 确保特征列的一致性
                for col in self.feature_names:
                    if col not in data.columns:
                        data[col] = 0
                data = data[list(self.feature_names) + (['hospital_death'] if 'hospital_death' in data.columns else [])]
            
            # 分离特征和标签
            y = data['hospital_death'].values if 'hospital_death' in data.columns else None
            X = data.drop(['hospital_death'], axis=1) if 'hospital_death' in data.columns else data
            
            # 标准化特征
            if not hasattr(self.scaler, 'mean_'):
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
            
            return X, y, max_id
            
        except Exception as e:
            print(f"数据处理出错: {str(e)}")
            raise
    
    def partial_fit(self, X=None, y=None, data=None, sample_size=None):
        """增量训练模型"""
        try:
            print("开始增量学习...")
            
            # 如果提供了原始数据，先处理数据
            if data is not None:
                X, y, max_id = self.process_data(data, sample_size)
            elif X is None or y is None:
                X, y, max_id = self.process_data(sample_size=sample_size)
            
            if X is None or y is None:
                return False
            
            # 增量训练
            self.model.partial_fit(X, y, classes=np.unique(y))
            
            # 评估模型
            predictions = self.model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y, predictions)
            accuracy = accuracy_score(y, predictions > 0.5)
            
            metrics = {
                'auc_score': float(auc_score),
                'accuracy': float(accuracy)
            }
            
            print(f"\n当前模型性能：")
            print(f"AUC Score: {auc_score:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # 存模型
            model_path = os.path.join(self.model_dir, "incremental_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'imputer': self.imputer
                }, f)
            
            # 更新训练日志
            if max_id is not None:
                self.save_training_log(max_id, metrics, len(y))
            
            print(f"\n模型已保存到: {model_path}")
            return True
            
        except Exception as e:
            print(f"增量学习过程中出错: {str(e)}")
            return False
    
    def predict(self, data):
        """对新数据进行预测"""
        try:
            # 如果输入是字典，转换为DataFrame
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # 删除不需要的列
            drop_columns = ['encounter_id', 'hospital_id', 'Unnamed: 83']
            data = data.drop(columns=[col for col in drop_columns if col in data.columns])
            
            # 处理输入数据
            X, _, _ = self.process_data(data)
            if X is None:
                raise ValueError("数据处理失败")
            
            # 预测死亡概率
            death_prob = self.model.predict_proba(X)[:, 1]
            return float(death_prob[0]) if isinstance(death_prob, np.ndarray) else float(death_prob)
            
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            raise
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_log.get('training_history', [])

def test_incremental_learning():
    """测试增量学习功能"""
    try:
        print("开始测试增量学习...")
        
        # 创建增量学习实例
        incremental = IncrementalLearning()
        
        # 测试不同样本数量的训练
        sample_sizes = [10, 50, 100]
        for size in sample_sizes:
            print(f"\n训练 {size} 个样本...")
            incremental.partial_fit(sample_size=size)
        
        # 测试预测功能
        print("\n测试预测功能...")
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
        df = pd.read_csv(csv_path)
        test_data = df.iloc[0].to_dict()
        
        # 删除不需要的列
        for col in ['encounter_id', 'hospital_id', 'Unnamed: 83']:
            if col in test_data:
                del test_data[col]
        
        death_prob = incremental.predict(test_data)
        print(f"预测死亡概率: {death_prob:.2%}")
        print(f"实际死亡结果: {test_data.get('hospital_death', 'unknown')}")
        
        # 打印训练历史
        print("\n训练历史:")
        for record in incremental.get_training_history():
            print(f"\n时间: {record['timestamp']}")
            print(f"样本数量: {record['samples_trained']}")
            print(f"ID范围: {record['id_range']}")
            print(f"AUC分数: {record['metrics']['auc_score']:.4f}")
            print(f"准确率: {record['metrics']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")

if __name__ == "__main__":
    test_incremental_learning() 