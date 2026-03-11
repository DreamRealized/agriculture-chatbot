from utils.models import generate_with_ollama, generate_with_ollama_stream
from utils.promptsArchive import (END_KEYWORD, get_mcp_prompt,
                                  get_summary_prompt)
from utils.utils import clean_message, extract_json, get_tables_by_keys, get_mcp_config_by_keyword

from .mcp_stream import call_tool_with_stream
import json

async def run_postgres_mcp_tool(user_query: str, context_list: list[str], rag_result: list[str]):
    context = ""
    times = 1
    config = get_mcp_config_by_keyword(user_query)
    db_key = list(config.keys())[0]
    db_config = config[db_key]
    dbs = get_tables_by_keys(user_query, db_config.get("keyword_maps"), db_config.get("schemas")) or config.get("schemas")
    print("dbs:-------------------------------------", dbs)


    yield "loading: mcp_begining \n\n"
    last_tool = ""
    last_tool_output = ""
    while True:
        # 在 LLM 规划前显示进度
        if times == 1:
            yield "loading: mcp_progress 正在理解您的问题, 制定查询计划... \n\n"
        else:
            yield f"loading: mcp_progress {getCompletionDesc(last_tool, times - 1, last_tool_output)} \n\n"

        prompt = get_mcp_prompt(user_query, context, dbs)
        llm_reply = await generate_with_ollama(prompt)
        llm_reply = clean_message(llm_reply["response"])

        if END_KEYWORD in llm_reply or times == 5:
            if times == 1:
                yield "loading: mcp_ending_not_match \n\n"
                break
            yield "loading: mcp_ending_summarize \n\n"
            summary_prompt = get_summary_prompt(user_query, context, rag_result)
            final_token = ""
            async for chunk in generate_with_ollama_stream(
                prompt=summary_prompt, model="qwen3:32b"
            ):
                token = (
                    str(chunk.get("response"))
                    if isinstance(chunk, dict) and "response" in chunk
                    else str(chunk)
                )
                print("token-------------", token)
                final_token += token
                yield token

            context_list.append(f"summary: {final_token}\n")
            break

        try:
            plan = extract_json(llm_reply)
            print("tool plan----------------------", plan)
            tool = plan["tool"]
            args = plan.get("args", {})

            # 显示带步骤号的工具执行进度
            last_tool = tool
            last_tool_output = ""
            yield f"loading: mcp_progress {getLoadingDescByTool(tool, times)} \n\n"
            async for tool_output in call_tool_with_stream(tool, db_key, args):
                last_tool_output += str(tool_output)
                context_list.append(
                    f"tool: {tool}, args: {args}, output: {tool_output}\n"
                )
                context += f"TOOL({tool}, {args}) => {tool_output}\n"
                print("tool_output------------------", tool_output)
            times += 1
        except Exception as e:
            print("error in run_agent-----------------", e)
            yield f"** Error parsing or executing plan: {e} **\n"
            break


# 生成带步骤号的中文进度描述（不暴露数据库敏感信息）
def getLoadingDescByTool(tool, step):
    match tool:
        case "list_schemas":
            return f"第{step}步: 正在获取数据结构信息..."
        case "list_objects":
            return f"第{step}步: 正在获取数据目录..."
        case "get_object_details":
            return f"第{step}步: 正在获取数据详情..."
        case "execute_sql":
            return f"第{step}步: 正在执行数据查询..."
        case _:
            return f"第{step}步: 正在执行操作..."


# 根据上一步工具类型和返回结果生成完成描述
def getCompletionDesc(tool, step, output):
    result_info = _analyze_output(tool, output)
    match tool:
        case "list_schemas":
            return f"第{step}步: 已获取数据结构信息, 正在进一步分析..."
        case "list_objects":
            return f"第{step}步: 已获取数据目录, 正在确定查询目标..."
        case "get_object_details":
            return f"第{step}步: 已获取数据详情, 正在生成查询方案..."
        case "execute_sql":
            return f"第{step}步: {result_info}"
        case _:
            return f"第{step}步: 操作完成, 正在分析结果..."


def _analyze_output(tool, output):
    """分析工具返回结果, 生成不含敏感信息的描述"""
    if not output or not output.strip():
        return "未获取到数据, 正在调整查询方案..."

    lower_output = output.lower()

    # 检测错误
    if "error" in lower_output or "exception" in lower_output:
        return "查询遇到问题, 正在修正查询方案..."

    # 尝试计算返回记录数
    row_count = _estimate_row_count(output)
    if row_count == 0:
        return "未查询到相关数据, 正在调整查询方案..."
    elif row_count > 0:
        return f"查询完成, 获取到 {row_count} 条相关记录, 正在分析数据..."

    return "查询完成, 正在分析返回的数据..."


def _estimate_row_count(output):
    """尝试从工具返回结果中估算记录数, 返回 -1 表示无法判断"""
    try:
        data = json.loads(output)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # 可能是 {"rows": [...]} 或 {"result": [...]} 格式
            for key in ("rows", "result", "data", "results"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
            return 1
    except (json.JSONDecodeError, TypeError):
        pass

    # 按行数估算（去掉空行）
    lines = [l for l in output.strip().split("\n") if l.strip()]
    if len(lines) > 1:
        return len(lines) - 1  # 减去可能的 header 行
    if len(lines) == 1:
        return 0 if "no " in lines[0].lower() or "empty" in lines[0].lower() else -1

    return -1
