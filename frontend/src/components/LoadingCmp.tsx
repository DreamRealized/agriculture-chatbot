import { useEffect, useState, useRef } from 'react'
import type { FC } from 'react'

export const extractLoadingInfo = (str: string, regex = /loading:\s*(.*?)\s*\n\n/g) => {
  // 匹配所有 loading: xxx
  const matches = [...str.matchAll(regex)];
  if (matches.length === 0) return { lastLoading: null, afterLast: "" };

  // 最后一个匹配
  const lastMatch = matches[matches.length - 1];
  const lastLoading = lastMatch[1];

  // 获取最后一个匹配之后的内容
  const afterLast = str.slice(lastMatch.index + lastMatch[0].length);

  return { lastLoading, afterLast };
}

// 脉冲圆点动画
const PulsingDots: FC = () => (
  <span className="inline-flex items-center ml-1 gap-[3px]">
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        className="inline-block w-[5px] h-[5px] rounded-full"
        style={{
          backgroundColor: '#22d3ee',
          animation: `dotPulse 1.4s ease-in-out ${i * 0.2}s infinite`,
        }}
      />
    ))}
  </span>
)

// 渐变流光进度条
const GradientBar: FC = () => (
  <div
    className="w-full h-[3px] rounded-full overflow-hidden mt-2"
    style={{ backgroundColor: 'rgba(255,255,255,0.1)' }}
  >
    <div
      className="h-full rounded-full"
      style={{
        width: '40%',
        background: 'linear-gradient(90deg, #22d3ee, #6366f1, #a855f7, #22d3ee)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s linear infinite',
      }}
    />
  </div>
)

export const LoadingCmp: FC<{ content: string }> = (prop) => {
  const [loadingText, setLoadingText] = useState('')
  const [fadeKey, setFadeKey] = useState(0)
  const prevTextRef = useRef('')

  useEffect(() => {
    const { content } = prop
    if (!content) {
      setLoadingText("小羲正在努力思考中, 请稍后")
      return
    }
    let text = ""
    const { lastLoading, afterLast } = extractLoadingInfo(content)
    if (afterLast) {
      setLoadingText("")
      return
    }
    switch (lastLoading) {
      case "mcp_begining":
        text = "小羲正在尝试使用MCP插件, 解决该问题"
        break
      case "mcp_list_schemas":
        text = "小羲正在分析数据库的架构, 请稍后"
        break
      case "mcp_list_objects":
        text = "小羲正在查询数据库的所有表, 请稍后"
        break
      case "mcp_get_object_details":
        text = "小羲正在查询数据表的表结构, 请稍后"
        break
      case "mcp_execute_sql":
        text = "小羲正在生成 SQL, 请稍后"
        break
      case "mcp_another_try":
        text = "小羲正在结合上次结果, 再次使用 MCP 插件, 请稍后"
        break
      case "mcp_ending_not_match":
        text = "尝试结束, 该问题不适合使用 MCP 工具解答"
        break
      case "mcp_ending_summarize":
        text = "尝试结束, 小羲正在汇总全部信息为您解答, 请稍后"
        break
      case "rag_analyzing":
        text = "请稍后, 小羲正在分析用户输入"
        break
      case "rag_no_relation":
        text = "用户输入分析完毕, 正在努力为您解答中"
        break
      case "rag_searching":
        text = "正在为您搜索相关资料中, 请稍后"
        break
      default:
        if (lastLoading?.startsWith("mcp_progress ")) {
          text = lastLoading.slice("mcp_progress ".length).replace(/\.{3}$/, '')
        }
        break
    }
    // 文案变化时触发淡入动画
    if (text !== prevTextRef.current) {
      prevTextRef.current = text
      setFadeKey(k => k + 1)
    }
    setLoadingText(text)
  }, [prop])

  if (!loadingText) return null

  return (
    <div className="py-1">
      <style>{`
        @keyframes dotPulse {
          0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1.2); }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(4px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
      <div
        key={fadeKey}
        className="text-sm"
        style={{
          color: '#94a3b8',
          animation: 'fadeSlideIn 0.3s ease-out',
        }}
      >
        {loadingText}
        <PulsingDots />
      </div>
      <GradientBar />
    </div>
  )
}
