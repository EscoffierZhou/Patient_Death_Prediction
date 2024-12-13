#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICU Patient Death Prediction System - Database Manager Module
数据管理模块，负责数据的存储、读取和处理

Copyright (c) 2023 Escoffier Zhou. All rights reserved.
This project is licensed under the MIT License. See LICENSE file for details.
"""

import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

@dataclass
class Result:
    success: bool
    message: str

class DatabaseManager:
    def __init__(self):
        # 修改为正确的 CSV 文件路径
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.csv_path = os.path.join(self.data_dir, "Patient.csv")
        print(f"CSV文件路径: {self.csv_path}")
    
    def get_all_patients(self) -> Tuple[List[Dict[str, Any]], str]:
        """获取所有病人信息"""
        try:
            # 使用 chunksize 分块读取大文件
            chunks = []
            for chunk in pd.read_csv(self.csv_path, encoding='utf-8-sig', chunksize=1000):
                chunks.append(chunk)
            df = pd.concat(chunks)
            print(f"成功读取到 {len(df)} 条记录")
            return df.to_dict('records'), "成功"
        except Exception as e:
            print(f"读取数据失败: {str(e)}")
            return [], f"读取数据失败: {str(e)}"
    
    def get_patient(self, patient_id: int) -> Tuple[Optional[Dict[str, Any]], str]:
        """获取指定ID的病人信息"""
        try:
            # ���用 chunksize 分块读取大文件
            for chunk in pd.read_csv(self.csv_path, encoding='utf-8-sig', chunksize=1000):
                patient = chunk[chunk['patient_id'] == patient_id].to_dict('records')
                if patient:
                    return patient[0], "成功"
            return None, "未找到病人"
        except Exception as e:
            print(f"读取数据失败: {str(e)}")
            return None, f"读取数据失败: {str(e)}"
    
    def add_patient(self, patient_data: Dict[str, Any]) -> Tuple[bool, str]:
        """添加新病人"""
        try:
            # 读取最后一个 chunk 来获取最大 ID
            max_id = 0
            for chunk in pd.read_csv(self.csv_path, encoding='utf-8-sig', chunksize=1000):
                chunk_max = chunk['patient_id'].max()
                if chunk_max > max_id:
                    max_id = chunk_max
            
            # 生成新的病人ID
            new_id = max_id + 1
            patient_data['patient_id'] = new_id
            
            # 添加新病人
            df = pd.DataFrame([patient_data])
            df.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            
            return True, f"病人添加成功，ID: {new_id}"
        except Exception as e:
            print(f"添加病人失败: {str(e)}")
            return False, f"添加病人失败: {str(e)}"
    
    def update_patient(self, patient_id: int, patient_data: Dict[str, Any]) -> Result:
        """更新病人信息"""
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            
            # 检查病人是否存在
            if not df[df['patient_id'] == patient_id].empty:
                # 更新数据
                df.loc[df['patient_id'] == patient_id] = patient_data
                df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                return Result(True, "病人信息更新成功")
            else:
                return Result(False, "未找到病人")
        except Exception as e:
            return Result(False, f"更新失败: {str(e)}")
    
    def delete_patient(self, patient_id: int) -> Tuple[bool, str]:
        """删除病人信息"""
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            
            # 检查病人是否存在
            if not df[df['patient_id'] == patient_id].empty:
                # 删除病人
                df = df[df['patient_id'] != patient_id]
                df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
                return True, "病人删除成功"
            else:
                return False, "未找到病人"
        except Exception as e:
            return False, f"删除失败: {str(e)}" 