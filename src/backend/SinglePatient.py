#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICU Patient Death Prediction System - Single Patient Module
病人信息处理模块，负责单个病人的数据处理和预测

Copyright (c) 2023 Escoffier Zhou. All rights reserved.
This project is licensed under the MIT License. See LICENSE file for details.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, Any
import joblib

class OperationResult:
    """操作结果类，用于返回操作状态和消息"""
    def __init__(self, success, message, data=None):
        self.success = success
        self.message = message
        self.data = data

class SinglePatient:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_list = None
        self.editing_cache = {}  # 用于缓存正在编辑的数据
        self.categorical_cols = [
            'ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 
            'icu_type', 'apache_2_bodysystem', 'apache_3j_bodysystem'
        ]
        
        # 定义字段验证规则和UI控件类型
        self.field_validators = {
            'age': {'type': 'int', 'min': 0, 'max': 130, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'bmi': {'type': 'float', 'min': 0, 'max': 100, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'elective_surgery': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'height': {'type': 'float', 'min': 0, 'max': 300, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'weight': {'type': 'float', 'min': 0, 'max': 500, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'pre_icu_los_days': {'type': 'float', 'min': 0, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'apache_2_diagnosis': {'type': 'int', 'min': 0, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'apache_3j_diagnosis': {'type': 'float', 'min': 0, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'apache_post_operative': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'arf_apache': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'gcs_eyes_apache': {'type': 'int', 'min': 0, 'max': 4, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'gcs_motor_apache': {'type': 'int', 'min': 0, 'max': 6, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'gcs_unable_apache': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'gcs_verbal_apache': {'type': 'int', 'min': 0, 'max': 5, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'heart_rate_apache': {'type': 'int', 'min': 0, 'max': 300, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'intubated_apache': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'map_apache': {'type': 'int', 'min': 0, 'max': 300, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'resprate_apache': {'type': 'int', 'min': 0, 'max': 100, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'temp_apache': {'type': 'float', 'min': 25, 'max': 45, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'ventilated_apache': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'hospital_death': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'aids': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'cirrhosis': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'diabetes_mellitus': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'hepatic_failure': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'immunosuppression': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'leukemia': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'lymphoma': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True},
            'solid_tumor_with_metastasis': {'type': 'int', 'min': 0, 'max': 1, 'ui_type': 'spinbox', 'default': None, 'allow_null': True}
        }
        
        # 定义分类字段的可选值和默认值
        self.categorical_values = {
            'ethnicity': {
                'values': ['African American', 'Asian', 'Caucasian', 'Hispanic', 
                          'Native American', 'Other/Unknown', 'Unknown'],
                'default': 'Unknown',
                'ui_type': 'combobox'
            },
            'gender': {
                'values': ['F', 'M', 'Unknown'],
                'default': 'Unknown',
                'ui_type': 'combobox'
            },
            'icu_admit_source': {
                'values': ['Accident & Emergency', 'Floor', 'Operating Room / Recovery',
                          'Other Hospital', 'Other ICU', 'Unknown'],
                'default': 'Unknown',
                'ui_type': 'combobox'
            },
            'icu_stay_type': {
                'values': ['admit', 'readmit', 'transfer'],
                'default': 'admit',
                'ui_type': 'combobox'
            },
            'icu_type': {
                'values': ['CCU-CTICU', 'CSICU', 'CTICU', 'Cardiac ICU', 'MICU',
                          'Med-Surg ICU', 'Neuro ICU', 'SICU'],
                'default': 'MICU',
                'ui_type': 'combobox'
            },
            'apache_2_bodysystem': {
                'values': ['Cardiovascular', 'Gastrointestinal', 'Haematologic',
                          'Metabolic', 'Neurologic', 'Renal/Genitourinary',
                          'Respiratory', 'Trauma', 'Undefined Diagnoses',
                          'Undefined diagnoses', 'Unknown'],
                'default': 'Unknown',
                'ui_type': 'combobox'
            },
            'apache_3j_bodysystem': {
                'values': ['Cardiovascular', 'Gastrointestinal', 'Genitourinary',
                          'Gynecological', 'Hematological', 'Metabolic',
                          'Musculoskeletal/Skin', 'Neurological', 'Respiratory',
                          'Sepsis', 'Trauma', 'Unknown'],
                'default': 'Unknown',
                'ui_type': 'combobox'
            }
        }
        
        # 特殊字段定义
        self.special_fields = {
            'patient_id': {
                'type': 'int',
                'ui_type': 'label',  # 只显示，不可编辑
                'editable': False,
                'default': None
            },
            'encounter_id': {
                'type': 'int',
                'ui_type': 'label',
                'editable': False,
                'default': None
            },
            'hospital_id': {
                'type': 'int',
                'ui_type': 'label',
                'editable': False,
                'default': None
            }
        }
        
        # 加载预处理器和模型
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        try:
            # 加载模型
            model_path = os.path.join(model_dir, "stacking_model.pkl")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("模型加载成功")
            
            # 加载预处理器
            with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(model_dir, "imputer.pkl"), 'rb') as f:
                self.imputer = pickle.load(f)
            with open(os.path.join(model_dir, "feature_list.pkl"), 'rb') as f:
                self.feature_list = pickle.load(f)
            print("预处理器和特征列表加载成功")
        except Exception as e:
            print(f"加载模型或预处理器时出错: {str(e)}")
    
    def get_field_info(self):
        """获取所有字段的信息（类型、UI控件类型和取值范围）
        Returns:
            dict: 字段信息字典
        """
        field_info = {}
        
        # 处理数值和布尔字段
        for field, validator in self.field_validators.items():
            field_info[field] = validator.copy()
        
        # 处理分类字段
        for field, info in self.categorical_values.items():
            field_info[field] = {
                'type': 'categorical',
                'ui_type': info['ui_type'],
                'values': info['values'],
                'default': info['default'],
                'editable': True
            }
        
        # 处理特殊字段
        for field, info in self.special_fields.items():
            field_info[field] = info.copy()
        
        return field_info
    
    def get_default_values(self):
        """获取所有字段的默认值
        Returns:
            dict: 字段默认值字典
        """
        defaults = {}
        
        # 获取数值和布尔字段的默认值
        for field, validator in self.field_validators.items():
            defaults[field] = validator['default']
        
        # 获取分类字段的默认值
        for field, info in self.categorical_values.items():
            defaults[field] = info['default']
        
        # 获取特殊字段的默认值
        for field, info in self.special_fields.items():
            defaults[field] = info['default']
        
        return defaults
    
    def validate_field(self, field_name, value):
        """验证字段值是否合法"""
        if field_name not in self.field_validators:
            return True, value

        validator = self.field_validators[field_name]
        
        # 如果允许空值且值为空，直接返回True
        if validator.get('allow_null', False) and (value is None or value == ''):
            return True, None

        field_type = validator['type']
        
        try:
            if field_type == 'int':
                if value is None or value == '':
                    return True, None
                value = int(value)
                if 'min' in validator and value < validator['min']:
                    return False, f"值必须大于等于 {validator['min']}"
                if 'max' in validator and value > validator['max']:
                    return False, f"值必须���于等于 {validator['max']}"
            elif field_type == 'float':
                if value is None or value == '':
                    return True, None
                value = float(value)
                if 'min' in validator and value < validator['min']:
                    return False, f"值必须大于等于 {validator['min']}"
                if 'max' in validator and value > validator['max']:
                    return False, f"值必须小于等于 {validator['max']}"
            elif field_type == 'str':
                if 'values' in validator and value not in validator['values']:
                    return False, f"值必须是以下之一: {', '.join(validator['values'])}"
        except (ValueError, TypeError):
            return False, "输入格式不正确"
        
        return True, value
    
    def get_patient_info(self, patient_id):
        """获取病人信息
        Args:
            patient_id (int): 病人ID
        Returns:
            dict: 病人信息字典
        """
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
            df = pd.read_csv(csv_path)
            
            patient_data = df[df['patient_id'] == int(patient_id)]
            if patient_data.empty:
                raise ValueError(f"未找到ID为{patient_id}的病人")
            
            # 转换数据类型
            data_dict = patient_data.iloc[0].to_dict()
            for field, value in data_dict.items():
                if pd.isna(value):
                    data_dict[field] = None
                elif field in self.field_validators:
                    validator = self.field_validators[field]
                    if validator['type'] == 'int':
                        data_dict[field] = int(value) if pd.notnull(value) else None
                    elif validator['type'] == 'str' and field in ['elective_surgery', 'apache_post_operative', 'arf_apache',
                                                                'gcs_unable_apache', 'intubated_apache', 'ventilated_apache',
                                                                'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
                                                                'immunosuppression', 'leukemia', 'lymphoma',
                                                                'solid_tumor_with_metastasis']:
                        # 对于布尔值字段，确保是字符串的'0'或'1'
                        if pd.notnull(value):
                            data_dict[field] = str(int(float(value)))
                        else:
                            data_dict[field] = None
            
            return data_dict
            
        except Exception as e:
            raise Exception(f"获取病人信息时出错: {str(e)}")
    
    def update_patient_info(self, patient_id, updated_data):
        """更新病人信息
        Args:
            patient_id (int): 病人ID
            updated_data (dict): 更新的数据字典
        Returns:
            OperationResult: 操作结果对象
        """
        try:
            # 验证所有字段
            for field, value in updated_data.items():
                is_valid, error_msg = self.validate_field(field, value)
                if not is_valid:
                    return OperationResult(False, f"更新失败：{error_msg}")
            
            # 读取CSV文件
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
            df = pd.read_csv(csv_path)
            
            # 找到对应的病人
            mask = df['patient_id'] == int(patient_id)
            if not any(mask):
                return OperationResult(False, f"更新失败：未找到ID为 {patient_id} 的病人")
            
            # 更新数据前进行类型转换
            converted_data = {}
            for field, value in updated_data.items():
                if field in self.field_validators:
                    validator = self.field_validators[field]
                    if value in ['', None]:
                        converted_data[field] = None
                    elif validator['type'] == 'int':
                        converted_data[field] = int(float(value))
                    elif validator['type'] == 'float':
                        converted_data[field] = float(value)
                    elif validator['type'] == 'str':
                        if field in ['elective_surgery', 'apache_post_operative', 'arf_apache',
                                   'gcs_unable_apache', 'intubated_apache', 'ventilated_apache',
                                   'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
                                   'immunosuppression', 'leukemia', 'lymphoma',
                                   'solid_tumor_with_metastasis']:
                            # 对于布尔值字段，确保是字符串的'0'或'1'
                            converted_data[field] = str(int(float(value))) if value not in ['', None] else None
                        else:
                            converted_data[field] = str(value)
                elif field in self.categorical_values:
                    converted_data[field] = str(value) if value not in ['', None] else None
                else:
                    converted_data[field] = value
            
            # 更新数据
            for field, value in converted_data.items():
                df.loc[mask, field] = value
            
            # 保存更新后的数据
            df.to_csv(csv_path, index=False)
            
            return OperationResult(True, f"更新成功：已更新ID为 {patient_id} 的病人信息", converted_data)
            
        except Exception as e:
            return OperationResult(False, f"更新失败：{str(e)}")
    
    def get_field_groups(self):
        """获取字段分组信息，用于在UI中组织字段
        Returns:
            list: 字段分组列表
        """
        return [
            {
                'name': '基本信息',
                'fields': ['patient_id', 'encounter_id', 'hospital_id', 'age', 'gender', 
                          'height', 'weight', 'bmi', 'ethnicity']
            },
            {
                'name': 'ICU信息',
                'fields': ['icu_id', 'icu_type', 'icu_admit_source', 'icu_stay_type', 
                          'pre_icu_los_days', 'elective_surgery']
            },
            {
                'name': 'Apache评分',
                'fields': ['apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative',
                          'apache_2_bodysystem', 'apache_3j_bodysystem', 'arf_apache',
                          'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache',
                          'gcs_unable_apache', 'heart_rate_apache', 'intubated_apache',
                          'map_apache', 'resprate_apache', 'temp_apache', 'ventilated_apache']
            },
            {
                'name': '每日生命体征',
                'fields': ['d1_diasbp_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_max',
                          'd1_diasbp_noninvasive_min', 'd1_heartrate_max', 'd1_heartrate_min',
                          'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max',
                          'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min',
                          'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max', 'd1_sysbp_min',
                          'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min',
                          'd1_temp_max', 'd1_temp_min']
            },
            {
                'name': '每小时生命体征',
                'fields': ['h1_diasbp_max', 'h1_diasbp_min', 'h1_diasbp_noninvasive_max',
                          'h1_diasbp_noninvasive_min', 'h1_heartrate_max', 'h1_heartrate_min',
                          'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max',
                          'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min',
                          'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min',
                          'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min']
            },
            {
                'name': '实验室检查',
                'fields': ['d1_glucose_max', 'd1_glucose_min', 'd1_potassium_max',
                          'd1_potassium_min']
            },
            {
                'name': '基础疾病',
                'fields': ['aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
                          'immunosuppression', 'leukemia', 'lymphoma',
                          'solid_tumor_with_metastasis']
            },
            {
                'name': '预后信息',
                'fields': ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob',
                          'hospital_death']
            }
        ]
    
    def get_max_patient_id(self):
        """获取当前最大的病人ID"""
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
            raw_data = pd.read_csv(csv_path)
            return raw_data['patient_id'].max()
        except Exception as e:
            print(f"获取最大病人ID时出错: {str(e)}")
            return None
    
    def add_new_patient(self, patient_data=None, patient_id=None):
        """添加新病人
        Args:
            patient_data (dict): 病人数据字典，包含所有必需的字段
            patient_id (int, optional): 指定的病人ID。如果为None，则使用最大ID+1
        Returns:
            OperationResult: 操作结果对象
        """
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
            
            # 读取现有数据
            df = pd.read_csv(csv_path)
            
            # 如果没有指定ID，使用最大ID+1
            if patient_id is None:
                patient_id = df['patient_id'].max() + 1
            elif patient_id in df['patient_id'].values:
                return OperationResult(False, f"添加失败：病人ID {patient_id} 已存在")
            
            # 如果没有提供病人数据，创建一个包含默认值的数据字典
            if patient_data is None:
                patient_data = {}
                
                # 添加特殊字段
                patient_data['patient_id'] = patient_id
                patient_data['encounter_id'] = df['encounter_id'].max() + 1
                patient_data['hospital_id'] = df['hospital_id'].iloc[0]
                
                # 添加分类字段的默认值
                for field, info in self.categorical_values.items():
                    patient_data[field] = info['default']
                
                # 添加数值字段的默认值
                for field, info in self.field_validators.items():
                    patient_data[field] = info['default']
            else:
                # 确保patient_id
                patient_data['patient_id'] = patient_id
            
            # 创建新的数据行
            new_row = pd.DataFrame([patient_data])
            
            # 确保所有必需的列都存在
            for col in df.columns:
                if col not in new_row.columns:
                    if col in self.categorical_values:
                        new_row[col] = self.categorical_values[col]['default']
                    elif col in self.field_validators:
                        new_row[col] = self.field_validators[col]['default']
                    else:
                        new_row[col] = 0
            
            # 按照原始数据的列顺序排列新数据
            new_row = new_row[df.columns]
            
            # 添加新行到数据文件
            df = pd.concat([df, new_row], ignore_index=True)
            
            # 保存更新后的数据
            df.to_csv(csv_path, index=False)
            
            # 初始化编辑缓存
            self.editing_cache[patient_id] = {
                'data': patient_data.copy(),
                'editing_field': None
            }
            
            return OperationResult(True, f"添加成功：已创建ID为 {patient_id} 的新病人", patient_data)
            
        except Exception as e:
            return OperationResult(False, f"添加失败：{str(e)}")
    
    def initialize_patient(self, patient_id):
        """初始化单病人的数据"""
        try:
            # 读取原始数据
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
            print(f"尝试读取CSV文件: {csv_path}")
            raw_data = pd.read_csv(csv_path)
            print(f"CSV文件读取成功，形状: {raw_data.shape}")
            
            # 获取指定病人的数据
            patient_data = raw_data[raw_data['patient_id'] == int(patient_id)].copy()
            if patient_data.empty:
                raise ValueError(f"未找到ID为{patient_id}的病人")
            print(f"找到病人数据，形状: {patient_data.shape}")
            
            # 删除不需要的列
            drop_columns = ['patient_id', 'encounter_id', 'hospital_id', 'Unnamed: 83']
            patient_data = patient_data.drop(columns=drop_columns, errors='ignore')
            
            # 获取数值列（不包括目标变量和分类变量）
            numeric_cols = [col for col in patient_data.columns 
                          if col not in self.categorical_cols 
                          and col != 'hospital_death']
            
            # 使用保存的imputer理缺
            if self.imputer is not None:
                patient_data[numeric_cols] = self.imputer.transform(patient_data[numeric_cols])
            
            # 使用保存的scaler进行标准化
            if self.scaler is not None:
                patient_data[numeric_cols] = self.scaler.transform(patient_data[numeric_cols])
            
            # 处理分类变量
            patient_data[self.categorical_cols] = patient_data[self.categorical_cols].fillna('Unknown')
            
            # 对分类变量进行独热编码
            patient_data = pd.get_dummies(patient_data, columns=self.categorical_cols)
            
            # 确保所有需要特征都
            if self.feature_list is not None:
                for feature in self.feature_list:
                    if feature not in patient_data.columns:
                        patient_data[feature] = 0
                # 按照训练时的特征顺序排列
                patient_data = patient_data[self.feature_list]
            
            print(f"特征数量: {len(patient_data.columns)}")
            print(f"特征列表: {list(patient_data.columns)}")
            
            return patient_data.values
            
        except Exception as e:
            raise Exception(f"数据处理过程中出错: {str(e)}")
        
    def preprocess_for_prediction(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """为预测准备数据"""
        try:
            print("\n开始数据预处理...")
            
            # 创建DataFrame
            df = pd.DataFrame([patient_data])
            print(f"原始数据形状: {df.shape}")
            
            # 处理0/1字段
            binary_fields = [
                'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
                'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis',
                'elective_surgery', 'apache_post_operative', 'arf_apache',
                'gcs_unable_apache', 'intubated_apache', 'ventilated_apache'
            ]
            for field in binary_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    print(f"二进制字段 {field} 的值: {df[field].iloc[0]}")
            
            # 删除不需要的列
            drop_columns = ['encounter_id', 'patient_id', 'hospital_id', 'Unnamed: 83', 'hospital_death']
            df = df.drop(columns=[col for col in drop_columns if col in df.columns])
            print(f"删除不需要的列后形状: {df.shape}")
            
            # 处理分类变量的缺值
            for col in self.categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown')
                    print(f"分类字段 {col} 的值: {df[col].iloc[0]}")
            
            # 对分类变量进行独热编码
            df = pd.get_dummies(df, columns=self.categorical_cols)
            print(f"独热编码后形状: {df.shape}")
            
            # 获取数值列
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            print(f"数值列数量: {len(numeric_cols)}")
            
            # 确保所有需要的数值列都存在
            missing_features = []
            for col in self.feature_list:
                if col not in df.columns:
                    missing_features.append(col)
                    df[col] = 0
            print(f"缺失的特征数量: {len(missing_features)}")
            if missing_features:
                print("缺失的特征:", missing_features[:10], "..." if len(missing_features) > 10 else "")
            
            # 使用imputer填充缺失值
            numeric_cols = [col for col in self.feature_list if col in numeric_cols]
            if numeric_cols:
                print("填充前的部分数值:", df[numeric_cols].iloc[0].head())
                df[numeric_cols] = self.imputer.transform(df[numeric_cols])
                print("填充后的部分数值:", df[numeric_cols].iloc[0].head())
                
                # 标准化数值特征
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
                print("标准化后的部分数值:", df[numeric_cols].iloc[0].head())
            
            # 确保所有特征都存在并按正确顺序排列
            final_df = pd.DataFrame(columns=self.feature_list)
            for col in self.feature_list:
                if col in df.columns:
                    final_df[col] = df[col]
                else:
                    final_df[col] = 0
            
            print(f"最终数据形状: {final_df.shape}")
            print("最终数据的前几个特征值:", final_df.iloc[0].head())
            
            return final_df
            
        except Exception as e:
            print(f"预处理失败: {str(e)}")
            raise
    
    def predict(self, patient_data: Dict[str, Any]) -> float:
        """预测病人死亡概率
        Args:
            patient_data (Dict[str, Any]): 病人数据
        Returns:
            float: 死亡概率
        """
        try:
            print("\n开始预测流程...")
            
            # 创建特征数据框
            df = pd.DataFrame([patient_data])
            
            # 删除不需要的列
            drop_columns = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id', 'hospital_death', 'Unnamed: 83']
            df = df.drop(columns=[col for col in drop_columns if col in df.columns])
            
            # 定义分类特征
            categorical_cols = [
                'ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 
                'icu_type', 'apache_2_bodysystem', 'apache_3j_bodysystem'
            ]
            
            # 处理分类变量的缺失值
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            
            # 对分类变量进行独热编码
            df = pd.get_dummies(df, columns=categorical_cols)
            
            # 获取数值列（不包括独热编码列）
            numeric_cols = [
                'age', 'bmi', 'elective_surgery', 'height', 'pre_icu_los_days', 'weight',
                'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative',
                'arf_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache',
                'gcs_verbal_apache', 'heart_rate_apache', 'intubated_apache', 'map_apache',
                'resprate_apache', 'temp_apache', 'ventilated_apache', 'd1_diasbp_max',
                'd1_diasbp_min', 'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min',
                'd1_heartrate_max', 'd1_heartrate_min', 'd1_mbp_max', 'd1_mbp_min',
                'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min', 'd1_resprate_max',
                'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max',
                'd1_sysbp_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min',
                'd1_temp_max', 'd1_temp_min', 'h1_diasbp_max', 'h1_diasbp_min',
                'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_heartrate_max',
                'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max',
                'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min',
                'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min',
                'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min', 'd1_glucose_max',
                'd1_glucose_min', 'd1_potassium_max', 'd1_potassium_min',
                'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'aids',
                'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression',
                'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
            ]
            
            # 确保所有数值列都存在
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # 使用imputer填充缺失值
            df[numeric_cols] = self.imputer.transform(df[numeric_cols].fillna(0))
            
            # 标准化数值特征
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
            # 确保所有特征都存在并按正确顺序排列
            final_df = pd.DataFrame(columns=self.feature_list)
            for col in self.feature_list:
                if col in df.columns:
                    final_df[col] = df[col]
                else:
                    final_df[col] = 0
            
            # 预测死亡概率
            death_prob = self.model.predict_proba(final_df)[0][1]
            print(f"预测完成，死亡概率: {death_prob:.4f}")
            
            return death_prob
            
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            raise
    
    def delete_patient(self, patient_id):
        """删除病人
        Args:
            patient_id (int): 病人ID
        Returns:
            OperationResult: 操作结果对象
        """
        try:
            # 读取CSV文件
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
            df = pd.read_csv(csv_path)
            
            # 找到对应的病人
            mask = df['patient_id'] == int(patient_id)
            if not any(mask):
                return OperationResult(False, f"删除失败：未找到ID为 {patient_id} 的病人")
            
            # 删除数据
            df = df[~mask]
            
            # 保存更新后的数据
            df.to_csv(csv_path, index=False)
            
            return OperationResult(True, f"删除成功：已删除ID为 {patient_id} 的病人")
            
        except Exception as e:
            return OperationResult(False, f"删除失败：{str(e)}")
    
    def start_field_edit(self, patient_id, field_name):
        """开始编辑某个字段
        Args:
            patient_id: 病人ID
            field_name: 字段名
        Returns:
            OperationResult: 包含当前和字段信息的操作结果
        """
        try:
            # 如果没有该病人的存，创建一个
            if patient_id not in self.editing_cache:
                # 获取病人当前数据
                patient_data = self.get_patient_info(patient_id)
                if patient_data is None:
                    return OperationResult(False, f"未找到ID为 {patient_id} 的病人")
                self.editing_cache[patient_id] = {
                    'data': patient_data,
                    'editing_field': None
                }
            
            # 获取字段信息
            field_info = self.get_field_info()[field_name]
            current_value = self.editing_cache[patient_id]['data'].get(field_name)
            
            # 更新正在编辑的字段
            self.editing_cache[patient_id]['editing_field'] = field_name
            
            return OperationResult(True, "开始编辑字段", {
                'field_info': field_info,
                'current_value': current_value
            })
            
        except Exception as e:
            return OperationResult(False, f"开始编辑段时出错: {str(e)}")
    
    def update_field_value(self, patient_id, field_name, value):
        """更新字段的值（临时保存在缓存中）
        Args:
            patient_id: 病人ID
            field_name: 字段名
            value: 新值
        Returns:
            OperationResult: 操作结果
        """
        try:
            if patient_id not in self.editing_cache:
                return OperationResult(False, "没有正在编辑的数据")
            
            # 验证字段值
            is_valid, error_msg = self.validate_field(field_name, value)
            if not is_valid:
                return OperationResult(False, error_msg)
            
            # 更新缓存中的值
            self.editing_cache[patient_id]['data'][field_name] = value
            
            return OperationResult(True, f"已更新字段 {field_name} 的值")
            
        except Exception as e:
            return OperationResult(False, f"更新字段值时出错: {str(e)}")
    
    def end_field_edit(self, patient_id, field_name, save=True):
        """结束字段编辑
        Args:
            patient_id: 病人ID
            field_name: 字段名
            save: 是否保存更改
        Returns:
            OperationResult: 操作结果
        """
        try:
            if patient_id not in self.editing_cache:
                return OperationResult(False, "没有正在编辑的数据")
            
            if save:
                # 获取要更新的数据
                update_data = {field_name: self.editing_cache[patient_id]['data'][field_name]}
                
                # 保存到数据库
                result = self.update_patient_info(patient_id, update_data)
                if not result.success:
                    return result
            
            # 清除正在编辑的字段标记
            self.editing_cache[patient_id]['editing_field'] = None
            
            return OperationResult(True, "已结束编辑" + (" 并保存更改" if save else ""))
            
        except Exception as e:
            return OperationResult(False, f"结束编辑时出错: {str(e)}")
    
    def get_editing_status(self, patient_id):
        """获取病人数据的编辑状态
        Args:
            patient_id: 病人ID
        Returns:
            dict: 包含编辑状态的字典
        """
        if patient_id not in self.editing_cache:
            return {
                'is_editing': False,
                'editing_field': None,
                'cached_data': None
            }
        
        cache = self.editing_cache[patient_id]
        return {
            'is_editing': cache['editing_field'] is not None,
            'editing_field': cache['editing_field'],
            'cached_data': cache['data']
        }
    
    def cancel_all_edits(self, patient_id):
        """取消所有未保存的编辑
        Args:
            patient_id: 病人ID
        Returns:
            OperationResult: 操作结果
        """
        try:
            if patient_id in self.editing_cache:
                del self.editing_cache[patient_id]
            return OperationResult(True, "已取消���有未保存的编辑")
        except Exception as e:
            return OperationResult(False, f"取消编辑时出错: {str(e)}")