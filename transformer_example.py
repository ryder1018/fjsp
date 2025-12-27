import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer

def create_sample_data(batch_size, seq_length, vocab_size):
    """創建範例訓練數據"""
    src = torch.randint(1, vocab_size, (batch_size, seq_length))
    tgt_input = torch.randint(1, vocab_size, (batch_size, seq_length))
    tgt_output = torch.randint(1, vocab_size, (batch_size, seq_length))
    return src, tgt_input, tgt_output

def train_step(model, src, tgt_input, tgt_output, optimizer, criterion):
    """單步訓練"""
    model.train()
    optimizer.zero_grad()
    
    output = model(src, tgt_input)
    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    # 模型參數
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    n_heads = 8
    n_layers = 4
    d_ff = 1024
    max_seq_length = 50
    dropout = 0.1
    
    # 訓練參數
    batch_size = 16
    seq_length = 20
    learning_rate = 0.0001
    num_epochs = 10
    
    # 創建模型
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, n_heads, 
        n_layers, d_ff, max_seq_length, dropout
    )
    
    # 優化器和損失函數
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練循環
    for epoch in range(num_epochs):
        # 創建批次數據
        src, tgt_input, tgt_output = create_sample_data(
            batch_size, seq_length, min(src_vocab_size, tgt_vocab_size)
        )
        
        # 訓練步驟
        loss = train_step(model, src, tgt_input, tgt_output, optimizer, criterion)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    print("訓練完成！")
    
    # 測試推理
    model.eval()
    with torch.no_grad():
        test_src = torch.randint(1, src_vocab_size, (1, 10))
        test_tgt = torch.randint(1, tgt_vocab_size, (1, 10))
        
        output = model(test_src, test_tgt)
        predicted = torch.argmax(output, dim=-1)
        
        print(f"輸入序列: {test_src[0].tolist()}")
        print(f"目標序列: {test_tgt[0].tolist()}")
        print(f"預測序列: {predicted[0].tolist()}")

if __name__ == "__main__":
    main()