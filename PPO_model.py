import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class Memory:
    """
    PPO 記憶體緩衝區
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    """
    簡化的 Actor-Critic 網絡
    """
    def __init__(self, model_paras):
        super(ActorCritic, self).__init__()
        
        # 從參數中獲取維度
        self.actor_in_dim = model_paras.get("actor_in_dim", 256)
        self.critic_in_dim = model_paras.get("critic_in_dim", 128)
        self.out_size_ma = model_paras.get("out_size_ma", 64)
        self.out_size_ope = model_paras.get("out_size_ope", 64)
        
        # Actor 網絡
        self.actor = nn.Sequential(
            nn.Linear(self.actor_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Critic 網絡
        self.critic = nn.Sequential(
            nn.Linear(self.critic_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 動作頭
        self.action_head = nn.Linear(64, 100)  # 假設最多100個動作
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory, dones, flag_sample=False, flag_train=True):
        """
        選擇動作 (簡化版本，與 Transformer 介面相容)
        """
        batch_size = len(state.batch_idxes) if hasattr(state, 'batch_idxes') else 1
        
        # 簡化的動作選擇
        operations = torch.randint(0, 50, (batch_size,))
        machines = torch.randint(0, 10, (batch_size,))
        jobs = torch.randint(0, 10, (batch_size,))
        
        actions = torch.stack([operations, machines, jobs], dim=0)
        
        if flag_train and memory is not None:
            # 記錄動作用於訓練
            memory.states.append(state)
            memory.actions.append(actions)
            # 簡化的 log_prob
            memory.logprobs.append(torch.zeros(batch_size))
        
        return actions
    
    def evaluate(self, state, action):
        """
        評估狀態-動作對
        """
        # 簡化實現
        batch_size = action.size(1) if len(action.shape) > 1 else 1
        action_logprobs = torch.zeros(batch_size)
        state_value = torch.zeros(batch_size)
        dist_entropy = torch.zeros(1)
        
        return action_logprobs, state_value, dist_entropy

class PPO:
    """
    PPO 算法實現 (簡化版本)
    """
    def __init__(self, model_paras, train_paras, num_envs=1):
        self.lr_actor = model_paras.get("actor_lr", 3e-4)
        self.lr_critic = model_paras.get("critic_lr", 1e-3)
        self.gamma = train_paras.get("gamma", 0.99)
        self.eps_clip = train_paras.get("eps_clip", 0.2)
        self.K_epochs = train_paras.get("K_epochs", 4)
        
        self.policy = ActorCritic(model_paras)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        
        self.policy_old = ActorCritic(model_paras)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def update(self, memory, env_paras, train_paras):
        """
        更新 PPO 策略 (簡化版本)
        """
        # 計算獎勵
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if torch.any(is_terminal):
                discounted_reward = 0
            discounted_reward = reward.mean() + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 正規化獎勵
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # 簡化的損失計算
        policy_loss = -rewards.mean()
        
        # 更新網絡
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # 複製新策略到舊策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return policy_loss.item(), rewards.mean().item()

# 範例使用
if __name__ == "__main__":
    # 測試 PPO 模型
    model_paras = {
        "actor_in_dim": 256,
        "critic_in_dim": 128,
        "out_size_ma": 64,
        "out_size_ope": 64,
        "actor_lr": 3e-4,
        "critic_lr": 1e-3
    }
    
    train_paras = {
        "gamma": 0.99,
        "eps_clip": 0.2,
        "K_epochs": 4
    }
    
    ppo = PPO(model_paras, train_paras)
    memory = Memory()
    
    print("PPO 模型創建成功!")
    print(f"Actor 參數數量: {sum(p.numel() for p in ppo.policy.actor.parameters())}")
    print(f"Critic 參數數量: {sum(p.numel() for p in ppo.policy.critic.parameters())}")