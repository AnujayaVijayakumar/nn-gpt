def calculate_meta_reward(current_score, baseline_score, valid_syntax):
    # 1. Syntax/Crash Penalty
    if not valid_syntax:
        print("   [RL] Penalty: Invalid Syntax/Crash.")
        return -1.0

    # 2. Improvement Reward
    delta = current_score - baseline_score

    if delta > 0:
        reward = delta * 1.0 
        # Cap to prevent exploding gradients
        reward = min(reward, 5.0) 
        print(f"   [RL] SUCCESS: +{delta:.2f} improvement. Reward: {reward:.4f}")
        return reward

    elif delta == 0:
        print(f"   [RL] Stagnation. Reward: -0.05")
        return -0.05

    else:
        # Penalize regression, but not as bad as a crash
        reward = max(delta * 0.1, -0.5)
        print(f"   [RL] Regression. Reward: {reward:.4f}")
        return reward