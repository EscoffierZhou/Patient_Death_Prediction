#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICU Patient Death Prediction System - Patient Detail Window Module
病人详情窗口模块，负责显示和编辑病人详细信息

Copyright (c) 2023 Escoffier Zhou. All rights reserved.
This project is licensed under the MIT License. See LICENSE file for details.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from DatabaseManager import DatabaseManager
from SinglePatient import SinglePatient
import pandas as pd

class PatientDetailWindow(tk.Toplevel):
    def __init__(self, parent, patient_data=None, is_new=False):
        super().__init__(parent)
        self.title("新建病人" if is_new else "病人详细信息")
        self.geometry("1200x800")
        
        self.parent = parent
        self.db = DatabaseManager()
        self.sp = SinglePatient()
        self.is_new = is_new
        
        # 获取CSV文件的列顺序
        try:
            # 读取CSV文件的第一行获取列顺序
            df = pd.read_csv(self.db.csv_path, nrows=0)
            self.csv_columns = list(df.columns)
            print(f"CSV列顺序: {self.csv_columns}")  # 调试信息
        except Exception as e:
            print(f"读取CSV列顺序失败: {str(e)}")
            self.csv_columns = []
        
        # 如果是新建病人，创建初始数据
        if is_new:
            # 获取所有病人ID并找到最大值
            patients, _ = self.db.get_all_patients()
            max_id = max([p.get('patient_id', 0) for p in patients], default=0)
            new_id = max_id + 1
            print(f"当前最大ID: {max_id}")  # 调试信息
            print(f"生成新ID: {new_id}")  # 调试信息
            
            # 创建初始数据字典，使用CSV列顺序
            self.patient_data = {}
            for col in self.csv_columns:
                if col == 'patient_id':
                    self.patient_data[col] = new_id
                else:
                    field_info = self.sp.get_field_info().get(col, {})
                    if field_info.get('type') == 'int':
                        self.patient_data[col] = 0
                    elif field_info.get('type') == 'float':
                        self.patient_data[col] = 0.0
                    elif field_info.get('type') == 'bool':
                        self.patient_data[col] = False
                    else:
                        self.patient_data[col] = ''
        else:
            # 如果是查看/编辑现有病人，按CSV列顺序重排数据
            ordered_data = {}
            for col in self.csv_columns:
                ordered_data[col] = patient_data.get(col, '')
            self.patient_data = ordered_data
        
        self.is_editing = is_new  # 如果是新建病人直接进入编辑模式
        self.entry_widgets = {}  # 存储输入框组件
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建画布和滚动条
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 配置网格列权重
        for i in range(5):  # 5列
            self.scrollable_frame.grid_columnconfigure(i, weight=1)
        
        # 显示数据
        self.display_data()
        
        # 布局
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 创建按钮框架
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        if is_new:
            # 新建病人模式：显��保存和取消按钮
            self.save_button = ttk.Button(button_frame, text="保存", command=self.save_changes)
            self.save_button.pack(side=tk.LEFT, padx=5)
            self.cancel_button = ttk.Button(button_frame, text="取消", command=self.cancel_edit)
            self.cancel_button.pack(side=tk.LEFT, padx=5)
        else:
            # 查看模式：显示修改和预测按钮
            self.edit_button = ttk.Button(button_frame, text="修改", command=self.toggle_edit_mode)
            self.edit_button.pack(side=tk.LEFT, padx=5)
            self.predict_button = ttk.Button(button_frame, text="预测死亡率", command=self.predict_death_rate)
            self.predict_button.pack(side=tk.LEFT, padx=5)
            
            # 添加删除按钮（使用红色突出显示）
            style = ttk.Style()
            style.configure("Delete.TButton", foreground="red")
            self.delete_button = ttk.Button(button_frame, text="删除病人", 
                                          command=self.delete_patient,
                                          style="Delete.TButton")
            self.delete_button.pack(side=tk.RIGHT, padx=5)  # 放在右侧
            
            # 创建但不显示保存和取消按钮
            self.save_button = ttk.Button(button_frame, text="保存", command=self.save_changes)
            self.cancel_button = ttk.Button(button_frame, text="取消", command=self.cancel_edit)
    
    def display_data(self):
        """显示病人数据"""
        # 清空现有内容
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # 使用CSV列顺序显示字段
        total_fields = len(self.csv_columns)
        fields_per_row = 5
        
        # 计算行数
        num_rows = (total_fields + fields_per_row - 1) // fields_per_row
        
        # 创建网格显示
        for row in range(num_rows):
            for col in range(fields_per_row):
                idx = row * fields_per_row + col
                if idx < total_fields:
                    field = self.csv_columns[idx]  # 使用CSV列顺序
                    # 获取字段类型信息
                    field_info = self.sp.get_field_info().get(field, {})
                    field_type = field_info.get('type', 'str')
                    
                    # 创建标签框架
                    frame = ttk.LabelFrame(self.scrollable_frame, text=f"{field} ({field_type})")
                    frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
                    
                    value = self.patient_data.get(field, '')
                    if isinstance(value, float) and np.isnan(value):
                        value = 'N/A'
                    elif value is None:
                        value = 'N/A'
                    
                    # 创建内部框架以实现居中对齐
                    inner_frame = ttk.Frame(frame)
                    inner_frame.pack(expand=True, fill="both", padx=5, pady=5)
                    
                    if self.is_editing and field != 'patient_id':  # ID字段不允许编辑
                        # 在编辑模式下创建输入框
                        entry = ttk.Entry(inner_frame, justify='center')
                        entry.insert(0, str(value))
                        
                        # 添加类型提示
                        if field_type == 'float':
                            tip_text = "请输入小数"
                            if 'min' in field_info and 'max' in field_info:
                                tip_text += f" ({field_info['min']} - {field_info['max']})"
                        elif field_type == 'int':
                            tip_text = "请输入整数"
                            if 'min' in field_info and 'max' in field_info:
                                tip_text += f" ({field_info['min']} - {field_info['max']})"
                        elif field_type == 'bool':
                            tip_text = "请输入 True 或 False"
                        else:
                            tip_text = "请输入文本"
                        
                        tip_label = ttk.Label(inner_frame, text=tip_text, font=('Arial', 8), foreground='gray')
                        
                        entry.pack(expand=True, fill="x")
                        tip_label.pack(expand=True, fill="x")
                        self.entry_widgets[field] = entry
                    else:
                        # 在查看模式下创建标签
                        if field_type == 'float' and isinstance(value, (int, float)):
                            # 浮点数保留2位小数显示
                            label = ttk.Label(inner_frame, text=f"{value:.2f}", wraplength=150, justify='center')
                        else:
                            label = ttk.Label(inner_frame, text=str(value), wraplength=150, justify='center')
                        label.pack(expand=True, fill="both")
                    
                    # 设置最小宽度以确保所有框架大小一致
                    frame.grid_propagate(False)
                    frame.configure(width=200, height=100)
    
    def toggle_edit_mode(self):
        """切换编辑模式"""
        self.is_editing = not self.is_editing
        if self.is_editing:
            # 进入编辑模式
            self.edit_button.pack_forget()
            self.predict_button.pack_forget()
            self.delete_button.pack_forget()  # 隐藏删除按钮
            self.save_button.pack(side=tk.LEFT, padx=5)
            self.cancel_button.pack(side=tk.LEFT, padx=5)
        else:
            # 退出编辑模式
            self.save_button.pack_forget()
            self.cancel_button.pack_forget()
            self.edit_button.pack(side=tk.LEFT, padx=5)
            self.predict_button.pack(side=tk.LEFT, padx=5)
            self.delete_button.pack(side=tk.RIGHT, padx=5)  # 恢复删除按钮
        
        # 重新显示数据
        self.display_data()
    
    def save_changes(self):
        """保存修改"""
        try:
            # 收集修改后的数据
            new_data = dict(self.patient_data)  # 创建原始数据的副本
            
            # 如果是新建病人，确保ID不被修改
            if self.is_new:
                original_id = self.patient_data['patient_id']
                print(f"保存时ID: {original_id}")  # 调试信息
            
            for field, entry in self.entry_widgets.items():
                if field == 'patient_id' and self.is_new:
                    # 新建病人时跳过ID字段的修改
                    continue
                    
                value = entry.get().strip()
                
                # 获取字段类型信息
                field_info = self.sp.get_field_info().get(field, {})
                field_type = field_info.get('type', 'str')
                
                # 转换值类型
                try:
                    if value == 'N/A':
                        new_data[field] = np.nan
                    elif field_type == 'int':
                        # 检查是否为有效的整数
                        try:
                            temp = float(value)
                            if temp != int(temp):
                                raise ValueError("必须是整数")
                            new_data[field] = int(temp)
                        except ValueError as e:
                            messagebox.showerror("错误", f"字段 '{field}' 必须是整数")
                            return
                    elif field_type == 'float':
                        try:
                            new_data[field] = float(value)
                        except ValueError:
                            messagebox.showerror("错误", f"字段 '{field}' 必须是小数")
                            return
                    elif field_type == 'bool':
                        if value.lower() not in ['true', 'false']:
                            messagebox.showerror("错误", f"字段 '{field}' 必须是 True 或 False")
                            return
                        new_data[field] = value.lower() == 'true'
                    else:
                        new_data[field] = str(value)
                        
                    # 检查数值范围
                    if field_type in ['int', 'float']:
                        num_value = new_data[field]
                        if 'min' in field_info and num_value < field_info['min']:
                            messagebox.showerror("错误", f"字段 '{field}' 的值不能小于 {field_info['min']}")
                            return
                        if 'max' in field_info and num_value > field_info['max']:
                            messagebox.showerror("错误", f"字段 '{field}' 的值不能大于 {field_info['max']}")
                            return
                        
                except ValueError:
                    messagebox.showerror("错误", f"字段 '{field}' 的值 '{value}' 无效")
                    return
                
                # 验证字段值
                is_valid, error_msg = self.sp.validate_field(field, new_data[field])
                if not is_valid:
                    messagebox.showerror("错误", f"字段 '{field}' 验证失败: {error_msg}")
                    return
            
            if self.is_new:
                # 添加新病人时确保使用正确的ID
                new_data['patient_id'] = self.patient_data['patient_id']
                print(f"最终保存的ID: {new_data['patient_id']}")  # 调试信息
                
                # 按照CSV列顺序重新排列数据
                if self.csv_columns:
                    ordered_data = {}
                    for col in self.csv_columns:
                        ordered_data[col] = new_data.get(col, '')
                    new_data = ordered_data
                    print(f"重排序后的数据键顺序: {list(new_data.keys())}")  # 调试信息
                
                # 添加新病人
                success, message = self.db.add_patient(new_data)
                if success:
                    messagebox.showinfo("成功", f"病人添加成功，ID: {new_data['patient_id']}")
                    # 刷新主窗口的病人列表
                    if hasattr(self.parent, 'refresh_patient_list'):
                        self.parent.refresh_patient_list()
                    self.destroy()  # 关闭窗口
                else:
                    messagebox.showerror("错误", message)
            else:
                # 更新现有病人
                result = self.sp.update_patient_info(self.patient_data['patient_id'], new_data)
                if result.success:
                    messagebox.showinfo("成功", result.message)
                    self.patient_data = new_data  # 更新本地数据
                    self.toggle_edit_mode()  # 退出编辑模式
                else:
                    messagebox.showerror("错误", result.message)
                
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def cancel_edit(self):
        """取消编辑"""
        if messagebox.askyesno("确认", "确定要取消修改吗？所有保存的更改都将丢失。"):
            if self.is_new:
                self.destroy()  # 如果是新建病人，直接关闭窗口
            else:
                self.toggle_edit_mode()  # 退出编辑模式
    
    def predict_death_rate(self):
        """预测病人死亡率"""
        try:
            # 创建SinglePatient实例
            predictor = SinglePatient()
            
            # 获取预测结果
            death_prob = predictor.predict(self.patient_data)
            
            # 显示预测结果
            messagebox.showinfo("预测结果", f"该病人的死亡概率为: {death_prob:.2%}")
            
        except Exception as e:
            messagebox.showerror("错误", f"预测过程出错: {str(e)}")
    
    def delete_patient(self):
        """删除病人"""
        # 显示确认对话框
        if messagebox.askyesno("确认删除", 
                             f"确定要删除ID为 {self.patient_data['patient_id']} 的病人吗？\n此操作不可撤销！",
                             icon='warning'):
            try:
                # 调用数据库删除函数
                success, message = self.db.delete_patient(self.patient_data['patient_id'])
                
                if success:
                    messagebox.showinfo("成功", message)
                    # 刷新主窗口的病人列表（如果主窗口有这个方法）
                    if hasattr(self.parent, 'refresh_patient_list'):
                        self.parent.refresh_patient_list()
                    # 关闭当前窗口
                    self.destroy()
                else:
                    messagebox.showerror("错误", message)
                    
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {str(e)}") 