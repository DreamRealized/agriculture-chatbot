import { useEffect, useState, useRef } from 'react'
import type { FC } from 'react'
import { useTranslation } from 'react-i18next'

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
          backgroundColor: '#71717a',
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
    style={{ backgroundColor: '#f0f0f2' }}
  >
    <div
      className="h-full rounded-full"
      style={{
        width: '40%',
        background: 'linear-gradient(90deg, #e4e4e7, #a1a1aa, #e4e4e7)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s linear infinite',
      }}
    />
  </div>
)

export const LoadingCmp: FC<{ content: string }> = (prop) => {
  const { t } = useTranslation()
  const [loadingText, setLoadingText] = useState('')
  const [fadeKey, setFadeKey] = useState(0)
  const prevTextRef = useRef('')

  useEffect(() => {
    const { content } = prop
    if (!content) {
      setLoadingText(t("chat.loading.thinking"))
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
        text = t("chat.loading.usingTools")
        break
      case "mcp_list_schemas":
        text = t("chat.loading.analyzingData")
        break
      case "mcp_list_objects":
        text = t("chat.loading.analyzingData")
        break
      case "mcp_get_object_details":
        text = t("chat.loading.analyzingData")
        break
      case "mcp_execute_sql":
        text = t("chat.loading.queryingData")
        break
      case "mcp_another_try":
        text = t("chat.loading.retryingTools")
        break
      case "mcp_ending_not_match":
        text = t("chat.loading.noMatchingTool")
        break
      case "mcp_ending_summarize":
        text = t("chat.loading.summarizing")
        break
      case "rag_analyzing":
        text = t("chat.loading.analyzingRequest")
        break
      case "rag_no_relation":
        text = t("chat.loading.thinking")
        break
      case "rag_searching":
        text = t("chat.loading.searching")
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
  }, [prop, t])

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
          color: '#71717a',
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
