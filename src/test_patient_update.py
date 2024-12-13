from backend.SinglePatient import SinglePatient

def test_update_patient():
    patient = SinglePatient()
    patient_id = 25312  # 使用第一个病人的ID
    
    # 准备测试数据
    test_data = {
        'age': 70,
        'bmi': 23.5,
        'elective_surgery': '1',  # 测试布尔值字段
        'height': 175.5,
        'weight': 72.5,
        'pre_icu_los_days': 1.5,
        'apache_2_diagnosis': 114,
        'apache_3j_diagnosis': 503,
        'apache_post_operative': '1',  # 测试布尔值字段
        'arf_apache': '1',  # 测试布尔值字段
        'gcs_eyes_apache': 4,
        'gcs_motor_apache': 5,
        'gcs_unable_apache': '1',  # 测试布尔值字段
        'gcs_verbal_apache': 3,
        'heart_rate_apache': 90,
        'intubated_apache': '1',  # 测试布尔值字段
        'map_apache': 60,
        'resprate_apache': 20,
        'temp_apache': 37.5,
        'ventilated_apache': '1',  # 测试布尔值字段
        'aids': '0',  # 测试布尔值字段
        'cirrhosis': '1',  # 测试布尔值字段
        'diabetes_mellitus': '1',  # 测试布尔值字段
        'hepatic_failure': '0',  # 测试布尔值字段
        'immunosuppression': '1',  # 测试布尔值字段
        'leukemia': '0',  # 测试布尔值字段
        'lymphoma': '1',  # 测试布尔值字段
        'solid_tumor_with_metastasis': '0',  # 测试布尔值字段
        'ethnicity': 'Asian',  # 测试分类字段
        'gender': 'F',  # 测试分类字段
        'icu_admit_source': 'Operating Room / Recovery',  # 测试分类字段
        'icu_stay_type': 'readmit',  # 测试分类字段
        'icu_type': 'MICU',  # 测试分类字段
        'apache_2_bodysystem': 'Respiratory',  # 测试分类字段
        'apache_3j_bodysystem': 'Respiratory'  # 测试分类字段
    }
    
    # 测试空值
    test_data_with_null = test_data.copy()
    test_data_with_null['age'] = ''
    test_data_with_null['bmi'] = None
    test_data_with_null['elective_surgery'] = ''
    
    # 更新数据
    print("测试更新所有字段...")
    result = patient.update_patient_info(patient_id, test_data)
    print(f"更新结果: {result.success}, 消息: {result.message}")
    
    # 验证更新结果
    print("\n验证更新后的数据...")
    updated_data = patient.get_patient_info(patient_id)
    for field, value in test_data.items():
        actual_value = str(updated_data.get(field))
        expected_value = str(value)
        if actual_value != expected_value:
            print(f"字段 {field} 不匹配: 期望值 = {expected_value}, 实际值 = {actual_value}")
    
    # 测试空值
    print("\n测试空值...")
    result = patient.update_patient_info(patient_id, test_data_with_null)
    print(f"空值更新结果: {result.success}, 消息: {result.message}")
    
    # 验证空值更新结果
    updated_data = patient.get_patient_info(patient_id)
    for field in ['age', 'bmi', 'elective_surgery']:
        value = updated_data.get(field)
        print(f"空值字段 {field} = {value}")

if __name__ == '__main__':
    test_update_patient() 