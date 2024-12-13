import unittest
import tkinter as tk
from tkinter import ttk
from EventMonitor import PatientInfoWindow
from DatabaseManager import DatabaseManager
from SinglePatient import SinglePatient

class TestPatientModification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.db = DatabaseManager()
        cls.sp = SinglePatient()
        
        # 获取一个测试病人的数据
        cls.test_patient_id = 4
        cls.test_patient_data, _ = cls.db.get_patient(cls.test_patient_id)
        
    def setUp(self):
        # 为每个测试创建新的PatientInfoWindow
        self.window = PatientInfoWindow(self.root, self.test_patient_data)
        self.window.start_edit()  # 进入编辑模式
        
    def tearDown(self):
        # 清理窗口
        self.window.destroy()
        
    def test_boolean_fields(self):
        """测试布尔字段的编辑"""
        boolean_fields = [
            'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
            'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
        ]
        
        for field in boolean_fields:
            with self.subTest(field=field):
                print(f"\n{'='*50}")
                print(f"测试布尔字段: {field}")
                try:
                    # 获取字段在表格中的列索引和位置信息
                    col_index = self.window.fields.index(field)
                    print(f"字段 {field} 在字段列表中的索引: {col_index}")
                    print(f"字段列表中该位置的实际字段: {self.window.fields[col_index]}")
                    
                    # 获取所有列的标题
                    headers = [self.window.tree.heading(col)['text'] for col in self.window.tree['columns']]
                    print(f"表格中的列标题: {headers[col_index]}")
                    
                    # 获取当前值
                    item = self.window.tree.get_children()[0]
                    current_values = self.window.tree.item(item)['values']
                    print(f"所有列的值: {current_values}")
                    current_value = current_values[col_index]
                    print(f"当前值 (索引 {col_index}): {current_value}")
                    
                    # 模拟更新值
                    new_value = not (current_value.lower() == 'true' if isinstance(current_value, str) else bool(current_value))
                    print(f"尝试将值更新为: {new_value}")
                    
                    # 获取实际的列标识符
                    column = f"#{col_index + 1}"
                    bbox = self.window.tree.bbox(item, column)
                    print(f"列 {column} 的位置信息: {bbox}")
                    
                    self.window.update_cell_value(item, col_index, field, new_value)
                    
                    # 验证更新后的值
                    updated_values = self.window.tree.item(item)['values']
                    print(f"更新后所有列的值: {updated_values}")
                    updated_value = updated_values[col_index]
                    print(f"更新后的值 (索引 {col_index}): {updated_value}")
                    
                    # 验证字段类型信息
                    field_info = self.window.field_info.get(field, {})
                    print(f"字段类型信息: {field_info}")
                    
                    self.assertEqual(str(new_value).lower(), updated_value.lower())
                except Exception as e:
                    print(f"测试字段 {field} 时出错: {str(e)}")
                    print(f"错误类型: {type(e).__name__}")
                    import traceback
                    print(f"错误堆栈:\n{traceback.format_exc()}")
                    raise
            
    def test_categorical_fields(self):
        """测试分类字段的编辑"""
        categorical_fields = {
            'ethnicity': ['Asian', 'African American', 'Caucasian', 'Hispanic', 'Native American', 'Other/Unknown'],
            'gender': ['M', 'F'],
            'icu_admit_source': ['Floor', 'Operating Room / Recovery', 'Accident & Emergency', 'Other Hospital', 'Other ICU'],
            'icu_stay_type': ['admit', 'readmit', 'transfer'],
            'icu_type': ['CCU-CTICU', 'CSICU', 'CTICU', 'MICU', 'SICU']
        }
        
        for field, categories in categorical_fields.items():
            with self.subTest(field=field):
                print(f"\n测试分类字段: {field}")
                try:
                    col_index = self.window.fields.index(field)
                    item = self.window.tree.get_children()[0]
                    
                    # 获取当前值
                    current_value = self.window.tree.item(item)['values'][col_index]
                    print(f"当前值: {current_value}")
                    
                    # 选择一个不同的类别
                    new_value = categories[0] if current_value != categories[0] else categories[1]
                    self.window.update_cell_value(item, col_index, field, new_value)
                    
                    # 验证更新后的值
                    updated_value = self.window.tree.item(item)['values'][col_index]
                    print(f"更新后的值: {updated_value}")
                    self.assertEqual(new_value, updated_value)
                except Exception as e:
                    print(f"测试字段 {field} 时出错: {str(e)}")
                    raise
            
    def test_numeric_fields(self):
        """测试数值字段的编辑"""
        print("\n预处理器和特征列表加载成功")
        
        # 测试整数和浮点数字段
        numeric_test_cases = {
            'age': {'type': 'int', 'test_value': 25},
            'bmi': {'type': 'float', 'test_value': 22.5},
            'height': {'type': 'float', 'test_value': 170.0},
            'weight': {'type': 'float', 'test_value': 65.0},
            'heart_rate_apache': {'type': 'int', 'test_value': 80},
            'map_apache': {'type': 'int', 'test_value': 90},
            'resprate_apache': {'type': 'int', 'test_value': 16},
            'temp_apache': {'type': 'float', 'test_value': 37.0}
        }
        
        for field_name, test_info in numeric_test_cases.items():
            with self.subTest(field=field_name):
                print(f"\n测试数值字段: {field_name} (类型: {test_info['type']})")
                try:
                    # 获取字段在表格中的列索引
                    col_index = self.window.fields.index(field_name)
                    item = self.window.tree.get_children()[0]
                    
                    # 获取当前值
                    current_value = self.window.tree.item(item)['values'][col_index]
                    print(f"当前值: {current_value}")
                    
                    # 更新值
                    test_value = test_info['test_value']
                    processed_value = self.window.update_cell_value(item, col_index, field_name, test_value)
                    
                    # 验证返回值
                    if test_info['type'] == 'int':
                        self.assertEqual(processed_value, test_value)  # 验证返回值是否为整数
                    else:
                        self.assertAlmostEqual(float(processed_value), test_value, places=1)  # 验证浮点数值
                    
                    # 验证显示值
                    updated_values = self.window.tree.item(item)['values']
                    display_value = updated_values[col_index]
                    if test_info['type'] == 'int':
                        self.assertEqual(display_value, str(test_value))
                    else:
                        self.assertAlmostEqual(float(display_value), test_value, places=1)
                        
                except Exception as e:
                    print(f"测试字段 {field_name} 时出错: {str(e)}")
                    raise
            
def run_tests():
    """运行所有测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatientModification)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    
if __name__ == '__main__':
    run_tests() 