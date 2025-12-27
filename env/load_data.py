import torch
import numpy as np

def nums_detec(lines):
    """
    從 FJSP 文件中檢測工作數量、機器數量和操作數量
    """
    first_line = lines[0].strip().split()
    num_jobs = int(first_line[0])
    num_mas = int(first_line[1])
    
    # 計算總操作數
    num_opes = 0
    for i in range(1, num_jobs + 1):
        if i < len(lines):
            job_line = lines[i].strip().split()
            if job_line:
                num_opes += int(job_line[0])
    
    return num_jobs, num_mas, num_opes

def load_fjs(lines, num_mas, max_opes):
    """
    載入 FJSP 數據並轉換為張量格式
    返回: [proc_times, ope_ma_adj, ope_pre_adj, ope_sub_adj, 
           opes_appertain, num_ope_biases, nums_ope, cal_cumul_adj]
    """
    num_jobs, _, num_opes = nums_detec(lines)
    
    # 初始化張量
    proc_times = torch.zeros(max_opes, num_mas)
    ope_ma_adj = torch.zeros(max_opes, num_mas)
    ope_pre_adj = torch.zeros(max_opes, max_opes)
    ope_sub_adj = torch.zeros(max_opes, max_opes)
    opes_appertain = torch.zeros(max_opes, dtype=torch.long)
    num_ope_biases = torch.zeros(num_jobs, dtype=torch.long)
    nums_ope = torch.zeros(num_jobs, dtype=torch.long)
    cal_cumul_adj = torch.zeros(max_opes, max_opes)
    
    current_ope = 0
    
    # 處理每個工作
    for job_id in range(num_jobs):
        if job_id + 1 >= len(lines):
            break
            
        job_line = lines[job_id + 1].strip().split()
        if not job_line:
            continue
            
        job_num_opes = int(job_line[0])
        nums_ope[job_id] = job_num_opes
        num_ope_biases[job_id] = current_ope
        
        # 處理工作中的每個操作
        line_idx = 1
        for op_id in range(job_num_opes):
            if line_idx >= len(job_line):
                break
                
            # 設置操作歸屬
            opes_appertain[current_ope] = job_id
            
            # 獲取可用機器數量
            num_machines = int(job_line[line_idx])
            line_idx += 1
            
            # 處理每台可用機器
            for _ in range(num_machines):
                if line_idx + 1 >= len(job_line):
                    break
                    
                machine_id = int(job_line[line_idx])
                proc_time = int(job_line[line_idx + 1])
                
                if machine_id < num_mas:
                    proc_times[current_ope, machine_id] = proc_time
                    ope_ma_adj[current_ope, machine_id] = 1
                
                line_idx += 2
            
            # 設置操作間的前後關係
            if op_id > 0:
                prev_ope = current_ope - 1
                ope_pre_adj[current_ope, prev_ope] = 1
                ope_sub_adj[prev_ope, current_ope] = 1
                cal_cumul_adj[prev_ope, current_ope] = 1
            
            current_ope += 1
    
    return [proc_times, ope_ma_adj, ope_pre_adj, ope_sub_adj,
            opes_appertain, num_ope_biases, nums_ope, cal_cumul_adj]

# 範例使用
if __name__ == "__main__":
    # 測試數據
    test_lines = [
        "2 3",
        "2 2 0 5 1 3 2 1 8 2 4",
        "3 1 2 6 3 0 2 1 4 2 0 7 1 9"
    ]
    
    num_jobs, num_mas, num_opes = nums_detec(test_lines)
    print(f"Jobs: {num_jobs}, Machines: {num_mas}, Operations: {num_opes}")
    
    data = load_fjs(test_lines, num_mas, num_opes)
    print("Data loaded successfully!")
    print(f"Processing times shape: {data[0].shape}")
    print(f"Operation-machine adjacency shape: {data[1].shape}")