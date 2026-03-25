  问题诊断与修复总结                                                                                                                                                                           
                                                                                                                                                                                                 
  根本原因                                                                                                                                                                                     
                                                                                                                                                                                                 
  Qwen2.5-VL的VisionZip评分层选择错误。                                                                                                                                                        

  target_block_idx = n_blocks - 2 = 30，但Qwen2.5-VL的fullatt_block_indexes = [7, 15, 23, 31]，block 30不在其中，使用的是窗口注意力（windowed attention），而非全局注意力。

  为什么会导致OCR任务灾难性失败？

  ┌───────────────────────────────────┬──────────────────────────────────────────────────────────┬────────────────────┐
  │            注意力类型             │                         评分机制                         │        影响        │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┼────────────────────┤
  │ Qwen2-VL（全块全局注意力）        │ 全局注意力分数 → 文字token在整图中被识别为高重要性       │ OCR正常工作        │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┼────────────────────┤
  │ Qwen2.5-VL block 30（窗口注意力） │ 局部窗口内的注意力分数 → 背景token在文字旁边反而得分更高 │ 保留背景、丢弃文字 │
  └───────────────────────────────────┴──────────────────────────────────────────────────────────┴────────────────────┘

  这解释了：
  - textvqa: 0.262（vs Qwen2-VL 0.806）— 文字区域token被丢弃
  - mmbench: 91.3%（表现正常）— 场景理解token局部显著性足够

  修复内容

  在4处将评分层改为最后一个fullatt block（block 31）：

  1. qwen2_5vl_model.py — qwen2_5vl_vision_tower_forward_visionzip（VisionZip塔）
  2. qwen2_5vl_model.py — qwen2_5vl_vision_tower_forward_prumerge_plus（Prumerge+塔）
  3. monkeypatch.py — replace_qwen2_5vl（visionzip + prumerge+）
  4. monkeypatch.py — replace_qwen3vl（visionzip + prumerge+）

  修复逻辑：
  fullatt_indexes = getattr(model.visual, 'fullatt_block_indexes', None)
  target_block_idx = fullatt_indexes[-1] if fullatt_indexes else n_blocks - 2

  使用fullatt_block_indexes[-1]（block 31）保证评分使用全局注意力，与Qwen2-VL等价。Qwen2-VL没有此属性，自动退回旧逻辑保持兼容性。