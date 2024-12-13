from backend.SinglePatient import SinglePatient
import pandas as pd
import numpy as np
import os

def test_prediction():
    # 初始化SinglePatient类
    patient = SinglePatient()
    
    # 读取CSV文件获取所有病人数据
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data", "Patient.csv")
    df = pd.read_csv(csv_path)
    
    # 选择不同情况的测试用例
    test_cases = {
        '高风险病人': df[df['hospital_death'] == 1].sample(n=2, random_state=42),
        '低风险病人': df[df['hospital_death'] == 0].sample(n=2, random_state=42),
        '随机病人': df.sample(n=2, random_state=42)
    }
    
    print("开始预测测试...")
    print("=" * 80)
    
    total_correct = 0
    total_cases = 0
    
    for case_type, patients in test_cases.items():
        print(f"\n测试 {case_type}:")
        print("-" * 40)
        
        for _, patient_data in patients.iterrows():
            try:
                patient_id = patient_data['patient_id']
                
                # 获取关键特征的值
                print(f"\n病人ID: {patient_id}")
                print("关键特征:")
                key_features = ['age', 'bmi', 'apache_2_diagnosis', 'apache_3j_diagnosis', 
                              'heart_rate_apache', 'temp_apache', 'elective_surgery',
                              'ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type', 
                              'icu_type', 'apache_2_bodysystem', 'apache_3j_bodysystem']
                for feature in key_features:
                    print(f"{feature}: {patient_data.get(feature)}")
                
                # 进行预测
                death_prob = patient.predict(patient_data.to_dict())
                actual_death = patient_data['hospital_death']
                
                # 评估预测结果
                predicted_death = 1 if death_prob > 0.5 else 0
                is_correct = predicted_death == actual_death
                total_correct += is_correct
                total_cases += 1
                
                print(f"预测死亡概率: {death_prob:.2%}")
                print(f"实际死亡结果: {actual_death}")
                print(f"预测结果: {'正确' if is_correct else '错误'}")
                print("-" * 40)
                
            except Exception as e:
                print(f"处理病人 {patient_id} 时出错: {str(e)}")
                continue
    
    # 输出总体准确率
    if total_cases > 0:
        accuracy = total_correct / total_cases
        print(f"\n总体准确率: {accuracy:.2%} ({total_correct}/{total_cases})")

if __name__ == '__main__':
    test_prediction() 