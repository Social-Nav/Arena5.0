---
name: multi-review
description: |
  多模型联合代码审查。同时启动 Gemini 3.1 Pro、DeepSeek V4 Pro、GPT-5.4、Kimi K2.6
  四个不同模型的 subagent 独立分析同一问题，最后汇总结论。适用于：根因分析、架构评审、
  复杂 bug 排查、避免单一模型陷入调试死循环。
user-invocable: true
---

# 多模型联合审查 (Multi-Model Review)

## 触发方式

用户输入 `/multi-review <问题描述>` 来触发。

## 执行流程

当用户触发此 skill 时，你必须严格按照以下步骤执行：

### Step 1: 理解问题

从用户消息中提取：
- **问题描述**：要分析的具体问题
- **相关文件**：用户提到的文件路径或代码位置
- **上下文**：任何额外的约束或背景

### Step 2: 并行启动 4 个 Reviewer Subagent

**关键：必须在同一个消息中并行调用 4 个 Agent tool，不能串行。**

使用 `Agent` tool，`subagent_type` 设为 `"general-purpose"`，但通过 prompt 指定使用对应的 reviewer subagent：

1. **@reviewer-gemini** — Gemini 3.1 Pro 视角
2. **@reviewer-deepseek** — DeepSeek V4 Pro 视角
3. **@reviewer-gpt** — GPT-5.4 视角
4. **@reviewer-kimi** — Kimi K2.6 视角

每个 subagent 的 prompt 必须包含完全相同的问题描述和上下文，格式如下：

```
[多模型联合审查任务]

请独立分析以下问题，不要参考其他模型的观点。

**问题描述：**
<用户的问题描述>

**相关文件/代码位置：**
<用户提到的文件路径>

**上下文/约束：**
<用户提供的额外上下文>

请按照你的 reviewer 系统指令中定义的格式输出结构化审查报告。
```

### Step 3: 等待全部 4 个 Subagent 返回

4 个 subagent 并行执行，等待全部完成后进入下一步。

### Step 4: 综合结论

汇总 4 个模型的审查结果，输出以下结构：

```
## 多模型联合审查结论

### 共识发现（≥3 个模型一致）
- 列出多个模型都指出的问题

### 分歧点
| 问题 | Gemini | DeepSeek | GPT-5.4 | Kimi |
|------|--------|----------|---------|------|
| ...  | ...    | ...      | ...     | ...  |

### 各模型独立发现
<details>
<summary>Gemini 3.1 Pro 审查报告</summary>
（完整报告内容）
</details>

<details>
<summary>DeepSeek V4 Pro 审查报告</summary>
（完整报告内容）
</details>

<details>
<summary>GPT-5.4 审查报告</summary>
（完整报告内容）
</details>

<details>
<summary>Kimi K2.6 审查报告</summary>
（完整报告内容）
</details>

### 综合建议
- 基于 4 个模型的共识和分歧，给出最终的行动建议
```

## 重要约束

- **必须并行调用**：4 个 Agent tool 调用必须在同一个消息中发出，不能串行等待
- **独立审查**：每个 subagent 的 prompt 中必须强调"独立分析，不要参考其他模型"
- **完整呈现**：综合结论中必须包含每个模型的完整报告（使用折叠块）
- **标注共识**：明确标注哪些发现是多个模型一致的，哪些是单一模型独有的
