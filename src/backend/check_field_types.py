import pandas as pd
import numpy as np
import os

def analyze_field_types():
    # 读取CSV文件
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Patient.csv")
    df = pd.read_csv(csv_path)
    
    # 分析每个字段
    field_info = {}
    for column in df.columns:
        # 获取非空值
        non_null_values = df[column].dropna()
        
        if len(non_null_values) == 0:
            field_info[column] = {
                'suggested_type': 'unknown',
                'unique_values': [],
                'has_decimals': False,
                'min_value': None,
                'max_value': None,
                'sample_values': []
            }
            continue
        
        # 检查是否只包含0和1
        is_binary = set(non_null_values.unique()) <= {0, 1, 0.0, 1.0}
        
        # 检查是否为数值型
        is_numeric = pd.to_numeric(non_null_values, errors='coerce').notna().all()
        
        if is_numeric:
            # 检查是否包含小数
            has_decimals = any(x % 1 != 0 for x in non_null_values if pd.notna(x))
            
            # 获取最大最小值
            min_value = non_null_values.min()
            max_value = non_null_values.max()
            
            if is_binary:
                suggested_type = 'int (binary)'
            elif has_decimals:
                suggested_type = 'float'
            else:
                suggested_type = 'int'
        else:
            suggested_type = 'str'
            has_decimals = False
            min_value = None
            max_value = None
        
        # 获取唯一值
        unique_values = sorted(non_null_values.unique())
        if len(unique_values) > 10:
            unique_values = unique_values[:10]  # 只显示前10个唯一值
        
        # 获取样本值
        sample_values = non_null_values.head(5).tolist()
        
        field_info[column] = {
            'suggested_type': suggested_type,
            'unique_values': unique_values,
            'has_decimals': has_decimals,
            'min_value': min_value,
            'max_value': max_value,
            'sample_values': sample_values
        }
    
    # 打印分析结果
    print("\n字段类型分析结果:")
    print("=" * 80)
    for field, info in field_info.items():
        print(f"\n字段名: {field}")
        print(f"建议类型: {info['suggested_type']}")
        if info['suggested_type'] in ['int', 'float', 'int (binary)']:
            print(f"最小值: {info['min_value']}")
            print(f"最大值: {info['max_value']}")
            print(f"包含小数: {info['has_decimals']}")
        if len(info['unique_values']) <= 10:
            print(f"唯一值: {info['unique_values']}")
        else:
            print(f"部分唯一值: {info['unique_values']} ...")
        print(f"样本值: {info['sample_values']}")
        print("-" * 40)

if __name__ == '__main__':
    analyze_field_types() 