import random
import numpy as np

class CaseGenerator:
    """
    FJSP 案例生成器
    """
    def __init__(self, num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None):
        self.num_jobs = num_jobs
        self.num_mas = num_mas
        self.opes_per_job_min = opes_per_job_min
        self.opes_per_job_max = opes_per_job_max
        self.nums_ope = nums_ope if nums_ope else [
            random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)
        ]
        
    def get_case(self, case_id):
        """
        生成一個 FJSP 案例
        返回格式: (case_lines, case_id)
        """
        lines = []
        
        # 第一行: 工作數量 機器數量
        lines.append(f"{self.num_jobs} {self.num_mas}\n")
        
        # 為每個工作生成操作
        for job_id in range(self.num_jobs):
            num_operations = self.nums_ope[job_id]
            job_line = f"{num_operations} "
            
            # 為每個操作生成機器選項和處理時間
            for op_id in range(num_operations):
                # 隨機選擇可用機器數量 (1 到 num_mas)
                num_available_machines = random.randint(1, self.num_mas)
                available_machines = random.sample(range(self.num_mas), num_available_machines)
                
                job_line += f"{num_available_machines} "
                
                # 為每台可用機器生成處理時間
                for machine_id in available_machines:
                    processing_time = random.randint(1, 20)  # 處理時間 1-20
                    job_line += f"{machine_id} {processing_time} "
            
            lines.append(job_line + "\n")
        
        return lines, case_id

class SimpleDataLoader:
    """
    簡單的數據載入器，用於批次生成案例
    """
    def __init__(self, case_generator, batch_size):
        self.case_generator = case_generator
        self.batch_size = batch_size
        
    def get_batch(self):
        """
        獲取一個批次的案例
        """
        batch = []
        for i in range(self.batch_size):
            case_lines, case_id = self.case_generator.get_case(i)
            batch.append(case_lines)
        return batch

# 範例使用
if __name__ == "__main__":
    # 創建案例生成器
    generator = CaseGenerator(
        num_jobs=5,
        num_mas=3,
        opes_per_job_min=2,
        opes_per_job_max=4
    )
    
    # 生成一個案例
    case_lines, case_id = generator.get_case(0)
    
    print("生成的 FJSP 案例:")
    for line in case_lines:
        print(line.strip())