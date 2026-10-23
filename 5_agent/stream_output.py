"""课件 8-2：用同一个客户查询 Agent 对比 7 种流式输出模式。

运行：conda run -n langchain python D:/grow/langchain/5_agent/stream_output.py

先改下方 STREAM_MODE，再运行，观察同一问题在不同模式下的输出。
工具数据是课件中的模拟数据；运行本文件会真实调用 .env 配置的模型。
本例使用项目当前版本支持的 version="v1"，与课件的返回格式一致。
"""

import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessageChunk, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from rich import print as rprint


# ==================== 1. 选择要观察的模式 ====================
# 单模式：每次只看一种，更容易辨认它的特点。
# 可选：values / updates / messages / tasks / debug / checkpoints / custom
STREAM_MODE = "updates"

# 多模式：取消下面某一行的注释，可在同一次 Agent 执行中接收多种事件。
# STREAM_MODE = ["tasks", "updates"]                  # 课件中的组合
# STREAM_MODE = ["updates", "messages", "custom"]     # 步骤 + 打字机 + 业务进度
# STREAM_MODE = ["values", "updates", "messages", "tasks", "debug", "checkpoints", "custom"]

MODE_GUIDE = {
    "values": "每步的完整状态：观察 messages 如何累计，旧消息也会再次出现。",
    "updates": "节点产生的状态增量：观察 model / tools 本次新增了哪些消息。",
    "messages": "消息和元数据：观察模型文字逐步出现、工具调用参数片段及完整工具结果。",
    "tasks": "任务开始和结束事件：用相同 id 对照 input 与 result / error。",
    "debug": "详细调试事件：观察 type、timestamp、step 和 payload。",
    "checkpoints": "检查点快照：观察 checkpoint_id、完整状态和下一步节点。",
    "custom": "工具主动发送的业务进度：观察查询过程，区别于工具最终 return。",
}


# ==================== 2. 沿用课件中的三个工具 ====================
# 为方便比较，所有模式都使用相同的工具、问题和模拟返回值。
# writer(...) 发送给 custom 流的订阅者；return 的结果才会成为 ToolMessage。
# get_stream_writer() 必须在 LangGraph 执行上下文内使用，本例由 Agent 调用工具。
@tool
def query_customer_data(customer_id: str) -> dict:
    """查询客户基本信息。

    Args:
        customer_id: 客户 ID，用于唯一标识客户。
    """
    writer = get_stream_writer()
    writer({"tool": "query_customer_data", "message": f"开始查询客户 {customer_id}"})
    data = {"name": "张三", "level": "VIP", "join_date": "2023-01-15"}
    writer({"tool": "query_customer_data", "message": "客户基本信息查询完成"})
    return data


@tool
def check_order_history(customer_id: str) -> dict:
    """查询客户订单历史。

    Args:
        customer_id: 客户 ID，用于唯一标识客户。
    """
    writer = get_stream_writer()
    writer({"tool": "check_order_history", "customer_id": customer_id, "processed": 0, "total": 15})
    # 模拟分三批处理 15 笔订单，故意留出短暂间隔，让 custom 的实时进度可见。
    # 这只是教学演示的耗时，不是模型生成延迟或真实数据库查询。
    for batch in range(1, 4):
        time.sleep(0.25)
        writer({"tool": "check_order_history", "processed": batch * 5, "total": 15})
    return {"total_orders": 15, "total_spent": 25800.00}


@tool
def get_current_promotions() -> dict:
    """获取当前可用促销活动，返回模拟的活动名称与有效日期。"""
    writer = get_stream_writer()
    writer({"tool": "get_current_promotions", "message": "开始查询可用优惠"})
    data = {
        "promotions": ["老用户优惠", "会员专属折扣"],
        "valid_until": "2027-01-31",
    }
    writer({"tool": "get_current_promotions", "message": "可用优惠查询完成"})
    return data


def build_agent():
    """沿用当前工程的百炼配置，并启用检查点以展示 checkpoints 模式。"""
    # 以脚本位置定位工程根目录，从其他工作目录运行也能找到同一份 .env。
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    if not os.getenv("BAILIAN_API_KEY") or not os.getenv("BAILIAN_API_URL"):
        raise ValueError("请在工程根目录 .env 中配置 BAILIAN_API_KEY 和 BAILIAN_API_URL。")

    model = init_chat_model(
        model="openai:qwen3.8-max",
        api_key=os.getenv("BAILIAN_API_KEY"),
        base_url=os.getenv("BAILIAN_API_URL"),
    )
    return create_agent(
        model=model,
        tools=[query_customer_data, check_order_history, get_current_promotions],
        system_prompt=(
            "你是客户服务助手。本例工具返回的是模拟数据，请使用中文回答。"
            "收到客户查询后，分别调用 query_customer_data、check_order_history、"
            "get_current_promotions，每个工具调用一次，再根据结果汇总客户信息、"
            "历史订单和可用优惠，不要自行编造工具结果。"
        ),
        # checkpoints 必须配合 checkpointer 和调用时的 thread_id 使用。
        # 内存保存器只在当前进程内保留状态；程序退出后不会保存到磁盘。
        checkpointer=InMemorySaver(),
    )


def print_messages(messages):
    """只展示消息的关键内容，避免 ID、token 用量等元数据淹没对比重点。"""
    for index, message in enumerate(messages, start=1):
        print(f"  {index}. {type(message).__name__}")
        if message.content:
            print(f"     内容：{message.content}")
        if getattr(message, "tool_calls", None):
            print("     工具调用：")
            rprint(message.tool_calls)


def show_stream(customer_service_agent, stream_mode):
    """消费真实流事件，并针对各模式展示最值得观察的字段。"""
    selected_modes = [stream_mode] if isinstance(stream_mode, str) else stream_mode
    for mode in selected_modes:
        print(f"{mode}: {MODE_GUIDE[mode]}")
    print("\n问题：查询客户 CUST123456 的个人信息、历史订单和可用优惠。")
    print("工具可能并行执行，事件先后顺序与数量以本次实际运行结果为准。\n")

    # 每次演示使用新的会话，避免上次累计的历史消息干扰模式对比。
    config = {"configurable": {"thread_id": f"stream-demo-{uuid4()}"}}
    text_line_open = False

    for event in customer_service_agent.stream(
        {"messages": [{"role": "user", "content": "查询客户 ID 为 CUST123456 的个人信息、历史订单和可用优惠"}]},
        config=config,
        stream_mode=stream_mode,
        version="v1",
    ):
        # v1：传字符串时直接返回该模式的 chunk；传列表时返回 (模式名, chunk)。
        # 多模式事件是元组，不是“以模式名为 key 的字典”。
        if isinstance(stream_mode, str):
            mode, chunk = stream_mode, event
        else:
            mode, chunk = event

        # 想查看某个模式的全部原始字段，可在这里临时加上 rprint(chunk)。
        if mode == "messages":
            # messages 的 chunk 本身又是二元组：(消息片段或消息对象, metadata)。
            # 片段不保证恰好对应一个字或一个 token，也可能只有工具调用信息。
            token, metadata = chunk
            # 当前项目版本也会发出完整 ToolMessage；它没有 tool_call_chunks。
            # 这只是工具最终返回的整条消息，工具内部进度仍应通过 custom 查看。
            if isinstance(token, ToolMessage):
                if text_line_open:
                    print()
                    text_line_open = False
                print(f"[messages] 节点={metadata['langgraph_node']}，完整工具结果，工具={token.name}：")
                print(token.content)
                continue
            if isinstance(token, AIMessageChunk) and token.tool_call_chunks:
                if text_line_open:
                    print()
                    text_line_open = False
                print(f"[messages] 节点={metadata['langgraph_node']}，工具调用参数片段：")
                rprint(token.tool_call_chunks)
            if token.text:
                if not text_line_open:
                    print(f"[messages] 节点={metadata['langgraph_node']}，模型文字：", end="", flush=True)
                # 不逐片段换行；flush=True 使片段到达后立即显示，形成打字机效果。
                print(token.text, end="", flush=True)
                text_line_open = True
            continue

        # 组合模式中，其他事件可能插入文字流，先换行让各类输出保持可读。
        if text_line_open:
            print()
            text_line_open = False

        if mode == "values":
            # chunk = {"messages": [用户消息, ...到当前步骤的全部消息]}。
            # 注意：这是本步完整状态，不是本步新增消息；旧消息会重复展示。
            print(f"[values] 完整状态，累计消息数={len(chunk['messages'])}")
            print_messages(chunk["messages"])

        elif mode == "updates":
            # chunk = {"model": {"messages": [...]}} 或 {"tools": {"messages": [...]}}。
            # 外层 key 是节点名，内层是节点写入的增量；不要当作累计历史。
            for node, update in chunk.items():
                print(f"[updates] 节点={node}，本次新增消息数={len(update['messages'])}")
                print_messages(update["messages"])

        elif mode == "tasks":
            # 开始事件有 input / triggers；结束事件有 result / error / interrupts。
            # tasks 不保证包含开始/结束时间戳；需要时间戳时观察 debug。
            # 未捕获的异常仍会直接抛出，失败时不保证能收到完整的结束事件。
            phase = "开始" if "input" in chunk else "结束"
            print(f"[tasks] {phase}，节点={chunk['name']}，任务 id={chunk['id']}")
            rprint(chunk)

        elif mode == "debug":
            # type 区分 task / task_result / checkpoint；payload 是详细事件数据。
            # 本例启用了 checkpointer，因此也会看到 checkpoint 调试事件。
            print(f"[debug] type={chunk['type']}，step={chunk['step']}，timestamp={chunk['timestamp']}")
            rprint(chunk["payload"])

        elif mode == "checkpoints":
            # 不仅有完整状态 values，还有恢复、追踪所需的 config / metadata / next。
            # 与 values 不同：这里关注的是“保存了哪个快照，接下来执行什么”。
            checkpoint_id = chunk["config"]["configurable"]["checkpoint_id"]
            print(f"[checkpoints] step={chunk['metadata']['step']}，checkpoint_id={checkpoint_id}")
            print(f"  下一步节点：{chunk['next']}，快照消息数={len(chunk['values'].get('messages', []))}")
            rprint({"config": chunk["config"], "parent_config": chunk["parent_config"]})
            print_messages(chunk["values"].get("messages", []))

        elif mode == "custom":
            # chunk 就是工具里 writer(...) 发出的对象，可为字典、字符串等。
            # 此模式不会自动输出最终回答，也不会自动把 return 变成进度事件。
            print("[custom] 工具主动发送的业务进度：")
            rprint(chunk)

        print("-" * 70)

    if text_line_open:
        print()


if __name__ == "__main__":
    customer_service_agent = build_agent()
    show_stream(customer_service_agent, STREAM_MODE)
