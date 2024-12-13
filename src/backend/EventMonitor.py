#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICU Patient Death Prediction System - Event Monitor Module
主界面模块，负责用户界面和事件处理

Copyright (c) 2023 Escoffier Zhou. All rights reserved.
This project is licensed under the MIT License. See LICENSE file for details.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
from DatabaseManager import DatabaseManager
from SinglePatient import SinglePatient
from PatientDetailWindow import PatientDetailWindow
from IncrementalLearning import IncrementalLearning
import os
import numpy as np
import sys
from io import StringIO

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = StringIO()
        
    def write(self, str):
        self.buffer.write(str)
        self.text_widget.insert(tk.END, str)
        self.text_widget.see(tk.END)
        
    def flush(self):
        self.buffer.flush()

class EventMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("病人信息管理系统")
        self.root.geometry("1200x800")
        
        # 数据库管理器
        self.db = DatabaseManager()
        
        # 增量学习模型
        self.incremental = IncrementalLearning()
        
        # 排序状态
        self.sort_column = None
        self.sort_reverse = False
        
        # 创建主界面
        self.create_main_interface()
        
    def create_main_interface(self):
        """创建主界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 上半部分（原有界面）
        upper_frame = ttk.Frame(main_frame)
        upper_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建搜索框架
        search_frame = ttk.LabelFrame(upper_frame, text="搜索条件")
        search_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # 创建ID快速搜索框架
        id_search_frame = ttk.Frame(search_frame)
        id_search_frame.pack(padx=5, pady=5, fill=tk.X)
        
        ttk.Label(id_search_frame, text="快速ID搜索:").pack(side=tk.LEFT, padx=5)
        self.quick_id_entry = ttk.Entry(id_search_frame)
        self.quick_id_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(id_search_frame, text="查看详情", command=self.quick_id_search).pack(side=tk.LEFT, padx=5)
        
        # 添加分隔线
        ttk.Separator(search_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
        
        # 创建搜索条件网格
        search_grid = ttk.Frame(search_frame)
        search_grid.pack(padx=5, pady=5, fill=tk.X)
        
        # 第一行搜索条件
        ttk.Label(search_grid, text="病人ID:").grid(row=0, column=0, padx=5, pady=5)
        self.id_entry = ttk.Entry(search_grid)
        self.id_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(search_grid, text="年龄范围:").grid(row=0, column=2, padx=5, pady=5)
        self.age_from = ttk.Entry(search_grid, width=8)
        self.age_from.grid(row=0, column=3, padx=(5,0), pady=5, sticky='w')
        ttk.Label(search_grid, text="至").grid(row=0, column=3, pady=5)
        self.age_to = ttk.Entry(search_grid, width=8)
        self.age_to.grid(row=0, column=3, padx=(50,5), pady=5, sticky='e')
        
        # 第二行搜索条件
        ttk.Label(search_grid, text="性别:").grid(row=1, column=0, padx=5, pady=5)
        self.gender_var = tk.StringVar()
        self.gender_combo = ttk.Combobox(search_grid, textvariable=self.gender_var, values=['全部', '男', '女'], state='readonly', width=17)
        self.gender_combo.current(0)
        self.gender_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # 搜索和重置按钮
        button_frame = ttk.Frame(search_frame)
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="搜索", command=self.search_patient).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="重置", command=self.reset_search).pack(side=tk.LEFT, padx=5)
        
        # 创建病人列表框架
        list_frame = ttk.LabelFrame(upper_frame, text="病人列表")
        list_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=10)
        
        # 创建表格
        columns = ('patient_id', 'age', 'gender', 'status')
        self.patient_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        # 设置列标题和宽度
        column_widths = {
            'patient_id': 100,
            'age': 100,
            'gender': 100,
            'status': 100
        }
        column_texts = {
            'patient_id': '病人ID',
            'age': '年龄',
            'gender': '性别',
            'status': '状态'
        }
        
        # 设置列标题和绑定点击事件
        for col in columns:
            self.patient_tree.heading(col, text=column_texts[col],
                                   command=lambda c=col: self.sort_treeview(c))
            self.patient_tree.column(col, width=column_widths[col], anchor='center')
        
        # 添加滚动条
        y_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.patient_tree.yview)
        x_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.patient_tree.xview)
        self.patient_tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # 布局
        self.patient_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定双击事件
        self.patient_tree.bind('<Double-1>', self.on_patient_select)
        
        # 功能按钮框架
        button_frame = ttk.Frame(upper_frame)
        button_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # 添加功能按钮
        ttk.Button(button_frame, text="新建病人", command=self.create_new_patient).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="刷新列表", command=self.refresh_patient_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="导出数据", command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # 初始加载病人列表
        self.refresh_patient_list()
        
        # 下半部分（增量学习界面）
        lower_frame = ttk.Frame(main_frame)
        lower_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建增量学习框架
        learning_frame = ttk.LabelFrame(lower_frame, text="增量学习")
        learning_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # 创建左右分栏
        left_frame = ttk.Frame(learning_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(learning_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：训练控制
        control_frame = ttk.LabelFrame(left_frame, text="训练控制")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 样本数量选择
        sample_frame = ttk.Frame(control_frame)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sample_frame, text="训练样本数量:").pack(side=tk.LEFT, padx=5)
        self.sample_size_var = tk.StringVar(value="50")
        sample_size_entry = ttk.Entry(sample_frame, textvariable=self.sample_size_var, width=10)
        sample_size_entry.pack(side=tk.LEFT, padx=5)
        
        # 训练按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="开始训练", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="查看训练历史", command=self.show_training_history).pack(side=tk.LEFT, padx=5)
        
        # 右侧：控制台输出
        console_frame = ttk.LabelFrame(right_frame, text="控制台输出")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=10)
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 重定向标准输出到控制台
        self.console_redirector = ConsoleRedirector(self.console_text)
        sys.stdout = self.console_redirector
    
    def reset_search(self):
        """重置搜索条件"""
        self.id_entry.delete(0, tk.END)
        self.age_from.delete(0, tk.END)
        self.age_to.delete(0, tk.END)
        self.gender_combo.current(0)
        self.refresh_patient_list()
    
    def quick_id_search(self):
        """快速ID搜索并显示详细信息"""
        patient_id = self.quick_id_entry.get().strip()
        
        if not patient_id:
            messagebox.showwarning("警告", "请输入病人ID")
            return
            
        try:
            # 尝试转换为整数
            patient_id = int(patient_id)
            
            # 查询病人信息
            patient, message = self.db.get_patient(patient_id)
            
            if patient:
                # 显示详细信息窗口
                PatientDetailWindow(self.root, patient)
            else:
                messagebox.showerror("错误", f"未找到ID为 {patient_id} 的病人")
                
        except ValueError:
            messagebox.showerror("错误", "病人ID必须是数字")
        except Exception as e:
            messagebox.showerror("错误", f"查询出错: {str(e)}")
    
    def search_patient(self, event=None):
        """搜索病人信息"""
        # 保存当前的排序状态
        current_sort_column = self.sort_column
        current_sort_reverse = self.sort_reverse
        
        # 获取搜索条件
        patient_id = self.id_entry.get().strip()
        age_from = self.age_from.get().strip()
        age_to = self.age_to.get().strip()
        gender = self.gender_var.get()
        
        # 清空现有列表
        for item in self.patient_tree.get_children():
            self.patient_tree.delete(item)
            
        try:
            # 查询病人信息
            patients, message = self.db.get_all_patients()
            if patients:
                # 根据条件过滤
                filtered_patients = []
                for patient in patients:
                    # 检查ID
                    if patient_id and str(patient.get('patient_id', '')) != patient_id:
                        continue
                    # 检查年龄范围
                    age = patient.get('age')
                    if age_from and age and float(age) < float(age_from):
                        continue
                    if age_to and age and float(age) > float(age_to):
                        continue
                    # 检查性别
                    if gender != '全部' and gender != patient.get('gender'):
                        continue
                    
                    filtered_patients.append(patient)
                
                # 显示结果
                if filtered_patients:
                    for patient in filtered_patients:
                        self.patient_tree.insert('', 'end', values=(
                            patient.get('patient_id', ''),
                            patient.get('age', ''),
                            patient.get('gender', ''),
                            patient.get('status', '存活')
                        ))
                else:
                    messagebox.showinfo("提示", "未找到匹配的病人信息")
            else:
                messagebox.showwarning("警告", message)
                
        except Exception as e:
            messagebox.showerror("错误", str(e))
        
        # 如果之前有排序，恢复排序
        if current_sort_column:
            self.sort_column = None  # 重置排序状态以确保重新排序
            self.sort_treeview(current_sort_column)
            if current_sort_reverse != self.sort_reverse:
                self.sort_treeview(current_sort_column)  # 如果需要反转，再次调用
    
    def export_data(self):
        """导出病人数据"""
        try:
            # 获取当前显示的病人列表
            data = []
            for item in self.patient_tree.get_children():
                values = self.patient_tree.item(item)['values']
                data.append({
                    'patient_id': values[0],
                    'age': values[1],
                    'gender': values[2],
                    'status': values[3]
                })
            
            if not data:
                messagebox.showwarning("警告", "没有数据可供导出")
                return
                
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            # 导出数据
            df = pd.DataFrame(data)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                df.to_excel(file_path, index=False)
                
            messagebox.showinfo("成功", "数据导出成功")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出数据失败: {str(e)}")
    
    def on_patient_select(self, event):
        """处理病人选择事件"""
        selected_item = self.patient_tree.selection()
        if not selected_item:
            return
            
        # 获取选中的病人ID
        patient_id = self.patient_tree.item(selected_item[0])['values'][0]
        
        # 查询完整的病人信息
        patient, message = self.db.get_patient(patient_id)
        if patient:
            PatientDetailWindow(self.root, patient)
        else:
            messagebox.showwarning("警告", message)
            
    def refresh_patient_list(self):
        """刷新病人列表"""
        # 保存当前的排序状态
        current_sort_column = self.sort_column
        current_sort_reverse = self.sort_reverse
        
        # 清空现有列表
        for item in self.patient_tree.get_children():
            self.patient_tree.delete(item)
            
        try:
            # 获取所有病人信息
            patients, message = self.db.get_all_patients()
            if patients:
                for patient in patients:
                    self.patient_tree.insert('', 'end', values=(
                        patient.get('patient_id', ''),
                        patient.get('age', ''),
                        patient.get('gender', ''),
                        patient.get('status', '存活')
                    ))
                
                # 如果之前有排序，恢复排序
                if current_sort_column:
                    self.sort_column = None  # 重置排序状态以确保重新排序
                    self.sort_treeview(current_sort_column)
                    if current_sort_reverse != self.sort_reverse:
                        self.sort_treeview(current_sort_column)  # 如果需要反转，再次调用
            else:
                messagebox.showwarning("警告", message)
            
        except Exception as e:
            messagebox.showerror("错误", f"刷新列表失败: {str(e)}")
            
    def create_new_patient(self):
        """创建新病人"""
        PatientDetailWindow(self.root, is_new=True)
        
    def sort_treeview(self, col):
        """对表格按列排序"""
        # 获取所有项目
        items = [(self.patient_tree.set(item, col), item) for item in self.patient_tree.get_children('')]
        
        # 如果点击的是当前排序列，则反转排序顺序
        if self.sort_column == col:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_reverse = False
            self.sort_column = col
        
        # 更新所有列的标题
        for column in self.patient_tree['columns']:
            if column == col:
                # 显示排序指示器
                direction = " ↓" if self.sort_reverse else " ↑"
                self.patient_tree.heading(column, text=self.patient_tree.heading(column)['text'].rstrip(' ↑↓') + direction)
            else:
                # 移除其他列的排序指示器
                self.patient_tree.heading(column, text=self.patient_tree.heading(column)['text'].rstrip(' ↑↓'))
        
        # 根据数据类型进行排序
        if col in ['patient_id', 'age']:
            # 数值排序
            items.sort(key=lambda x: float(x[0]) if x[0] != '' else float('-inf'), reverse=self.sort_reverse)
        else:
            # 字符串排序
            items.sort(key=lambda x: x[0], reverse=self.sort_reverse)
        
        # 重新排列项目
        for index, (val, item) in enumerate(items):
            self.patient_tree.move(item, '', index)
    
    def start_training(self):
        """开始增量学习训练"""
        try:
            sample_size = int(self.sample_size_var.get())
            if sample_size <= 0:
                messagebox.showerror("错误", "样本数量必须大于0")
                return
            
            # 清空控制台
            self.console_text.delete(1.0, tk.END)
            
            # ���始训练
            success = self.incremental.partial_fit(sample_size=sample_size)
            
            if success:
                messagebox.showinfo("成功", "训练完成")
            else:
                messagebox.showwarning("警告", "没有新的数据可供训练")
                
        except ValueError:
            messagebox.showerror("错误", "样本数量必须是整数")
        except Exception as e:
            messagebox.showerror("错误", f"训练过程出错: {str(e)}")
    
    def show_training_history(self):
        """显示训练历史"""
        history_window = tk.Toplevel(self.root)
        history_window.title("训练历史")
        history_window.geometry("600x400")
        
        # 创建表格
        columns = ('timestamp', 'samples', 'id_range', 'auc', 'accuracy')
        tree = ttk.Treeview(history_window, columns=columns, show='headings')
        
        # 设置列标题
        tree.heading('timestamp', text='时间')
        tree.heading('samples', text='样本数量')
        tree.heading('id_range', text='ID范围')
        tree.heading('auc', text='AUC分数')
        tree.heading('accuracy', text='准确率')
        
        # 设置列宽
        for col in columns:
            tree.column(col, width=100)
        
        # 添加���动条
        scrollbar = ttk.Scrollbar(history_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充数据
        history = self.incremental.get_training_history()
        for record in history:
            tree.insert('', 'end', values=(
                record['timestamp'],
                record['samples_trained'],
                record['id_range'],
                f"{record['metrics']['auc_score']:.4f}",
                f"{record['metrics']['accuracy']:.4f}"
            ))
    
    def run(self):
        """运行事件监控器"""
        self.root.mainloop()

if __name__ == "__main__":
    monitor = EventMonitor()
    monitor.run() 