import React, { useState, useRef, useEffect, useMemo } from "react";
import type { Message } from "@/types/types";
import { generateResponse } from "@/lib/ollamaService";
import { ImageUploader } from "./ImageUploader";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { useTranslation } from "react-i18next";
import chatBg from "../assets/chat-bg.png";
import botAvatar from "../assets/bot-avatar.png";
import textBg from "../assets/text-bg.png";
import sendBg from "../assets/send-bg.png";
import { saveUserMessage, saveAIResponse, getPromptVersion } from "@/lib/managementApi";
import { logError } from "@/lib/logService";
import { SpeechToText } from "./SpeechToText.tsx"
import { CleanContextBtn } from "./CleanContextBtn.tsx";
import { useAudioPlayer } from "@/hooks/useAudioPlayer.ts";


interface ChatInterfaceProps {
  defaultModel: string;
  multimodalModel: string;
}

export const ChatInterface = ({
  defaultModel = "deepseek-r1:8b",
  multimodalModel = "qwen:2.5vl:7b"
}: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  // Add this new state for tracking image reset
  const [shouldResetImages, setShouldResetImages] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();
  const savedMessagesRef = useRef<Set<string>>(new Set());
  const [targetDB, setTargetDB] = useState("");
  const conversationId = useMemo(() => `chat_${Date.now()}`, [])
  const { addToAudioQueue, flushBuffer, clearQueue } = useAudioPlayer()

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 简单的token估算函数
  const estimateTokens = (text: string): number => {
    // 简单估算：中文字符 * 1.5 + 英文单词 * 1
    const chineseChars = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
    const englishWords = (text.match(/[a-zA-Z]+/g) || []).length;
    return Math.ceil(chineseChars * 1.5 + englishWords);
  };

  const sendMessage = async () => {
    if (!inputRef.current) return;
    const input = inputRef.current.value;
    if (!input.trim()) return;

    // 强制清空缓存区所有音频
    clearQueue();

    const userMessage: Message = {
      text: input,
      sender: "user",
      images: images.length > 0 ? [...images] : undefined
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Initialize AI message
    setMessages(prev => [
      ...prev,
      {
        text: "",
        sender: "ai",
        isComplete: false
      }
    ]);

    // Determine which model to use
    const modelToUse = userMessage.images ? multimodalModel : defaultModel;

    try {
      // 获取prompt版本
      const promptVersion = await getPromptVersion(modelToUse);

      // 保存用户消息到管理后台
      const startTime = Date.now();
      const savedUserMessage = await saveUserMessage(input, modelToUse, promptVersion);
      const userMessageId = savedUserMessage?.message_id;

      if (!userMessageId) {
        const error = new Error("保存用户消息失败");
        logError("保存用户消息失败", error, {
          model: modelToUse,
          message_length: input.length
        });
      }

      abortControllerRef.current = new AbortController();

      await generateResponse(
        {
          prompt: input,                    // ✅ 发送原始用户输入
          images: userMessage.images,      // ✅ 发送图片
          chat_id: conversationId, // Keep backend memory stable across message IDs.
          targetDB: targetDB
        },
        (data) => {
          if (data.type == 'delta') {
            setMessages(prev => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              updated[lastIdx] = {
                ...updated[lastIdx],
                text: (updated[lastIdx].text + data.token),
              };
              return updated;
            });
            addToAudioQueue(data.token)
          } else if (data.type == "done" && userMessage.images) {
            setShouldResetImages(true);
            setTargetDB("");
          }
        },
        error => {
          console.error("Error:", error);
          setMessages(prev => [
            ...prev,
            {
              text: `Error: ${error.message}`,
              sender: "ai",
              isComplete: true
            }
          ]);
        },
        abortControllerRef.current?.signal
      );

      flushBuffer(); // 强制清空缓存区所有音频

      // Mark as complete and save AI response
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        updated[lastIdx] = {
          ...updated[lastIdx],
          isComplete: true
        };

        // 保存AI回答到管理后台
        if (userMessageId && !savedMessagesRef.current.has(userMessageId)) {
          savedMessagesRef.current.add(userMessageId);
          const endTime = Date.now();
          const responseTime = (endTime - startTime) / 1000;
          const aiResponse = updated[lastIdx]?.text || '';
          const estimatedTokens = estimateTokens(aiResponse);

          // 异步保存AI回答
          saveAIResponse(
            userMessageId,
            aiResponse,
            modelToUse,
            responseTime,
            estimatedTokens,
            promptVersion
          ).catch(error => {
            console.error('保存AI回答失败:', error);
          });
        }
        return updated;
      });

    } catch (error) {
      console.error("发送消息失败:", error);
      setMessages(prev => [
        ...prev.slice(0, -1), // 移除未完成的AI消息
        {
          text: `发送失败: ${error instanceof Error ? error.message : '未知错误'}`,
          sender: "ai",
          isComplete: true
        }
      ]);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  // Add this useEffect to handle image reset
  useEffect(() => {
    if (shouldResetImages) {
      setImages([]);
      setShouldResetImages(false);
    }
  }, [shouldResetImages]);

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
    }
    const lastMsg = messages[messages.length - 1];
    if (lastMsg.sender === "ai" && lastMsg.text === "") {
      setMessages(prev => [
        ...prev.slice(0, -1), // 移除未完成的AI消息
        {
          text: `请求已中断, 欢迎再次向我提问.`,
          sender: "ai",
          isComplete: true
        }
      ]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // 使用React.memo + 自定义arePropsEqual
  const MemoizedMessage = React.memo(({ msg }: { msg: Message }) => (
    <div className={`mb-6 ${msg.sender === "user" ? "flex justify-end" : "flex justify-start"}`}>
      {msg.sender === "ai" && (
        <img src={botAvatar} className="w-10 h-10 mr-2" alt="AI" />
      )}
      <div
        style={{
          background: "rgba(17, 96, 73, 0.40)",
          borderRadius: "4px",
          padding: "10px",
          maxWidth: "80%",
          color: "#D7ECFF"
        }}
      >
        <MarkdownRenderer content={msg.text} />
      </div>
    </div>
  ), (prev, next) => {
    // 深度比较（假设msg是immutable）
    return prev.msg.text === next.msg.text
      && prev.msg.sender === next.msg.sender;
  });

  const cleanContext = React.useCallback(() => {
    setMessages([])
  }, [])

  return (
    <div className="w-screen bg-[#1a1a1a] min-h-screen flex items-center justify-center">
      <div
        className="relative w-full h-[100vh] sm:h-[747px] flex flex-col max-w-[680px]"
        style={{
          backgroundImage: `url(${chatBg})`,
          backgroundSize: "100% 100%",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat"
        }}
      >
        {/* <img */}
        {/*   src={robotPng} */}
        {/*   className="absolute bottom-[0] left-[-200px] w-[180px]" */}
        {/* /> */}
        <div className="w-full h-[45px] flex items-center justify-between px-8 flex-shrink-0">
          <span className="chat-title">AI智能决策</span>
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-2 mx-2">
          <div className="flex items-start gap-3 text-gray-400">
            <img src={botAvatar} className="w-10 h-10" alt="AI" />
            <div
              className="mb-6"
              style={{
                background: "rgba(17, 96, 73, 0.40)",
                borderRadius: "4px",
                padding: "10px",
                maxWidth: "80%",
                color: "#D7ECFF"
              }}
            >
              您好啊！！我是
              <span style={{ color: "#d6a130" }}>
                人工智能农业模拟器--"小羲"
              </span>
              。我可以为您解答相关农业方面农技、农艺问题，您可以在下方输入框内输入想要提问的问题！！！也可以点击麦克风语音识别！！！很高兴为您服务！！！
            </div>
          </div>
          {messages.map((msg, i) => (
            <MemoizedMessage key={i} msg={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div
          className="flex flex-col gap-2 m-4 flex-shrink-0"
          style={{
            backgroundImage: `url(${textBg})`,
            backgroundSize: "100% 100%",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
            padding: "8px 4px"
          }}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={t(
              images.length > 0 ? "chat.file_placeholder" : "chat.placeholder"
            )}
            className="w-full p-3 bg-transparent text-white placeholder-gray-400 focus:outline-none resize-none"
            disabled={isLoading}
            rows={2}
            style={{
              opacity: 1,
              pointerEvents: "auto"
            }}
          />

          <div className="flex items-center justify-end gap-2 mt-2">
            <ImageUploader
              onImagesChange={imgs => {
                setImages(imgs);
              }}
              disabled={isLoading}
              resetTrigger={shouldResetImages}
            />
            <div
              className="w-8 h-8 flex items-center justify-center cursor-pointer"
            >
              <SpeechToText
                disabled={isLoading}
                afterTranslate={(str) => {
                  setInput((prev) => {
                    return prev + str
                  })
                }}
                onVoiceInActive={() => {
                  const timeout = setTimeout(() => {
                    sendMessage();
                    clearTimeout(timeout);
                  }, 500);
                }}
              ></SpeechToText>
            </div>

            <div
              className="w-8 h-8 flex items-center justify-center cursor-pointer text-white"
            >
              <CleanContextBtn cleanContext={cleanContext} chatId={conversationId} />
            </div>

            <div
              className="relative"
              style={{
                backgroundImage: `url(${sendBg})`,
                backgroundSize: "100% 100%",
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat"
              }}
            >
              <div
                onClick={
                  isLoading
                    ? stopGeneration
                    : !input.trim()
                      ? undefined
                      : sendMessage
                }
                className={`px-4 h-8 font-medium cursor-pointer ${isLoading
                  ? "text-red-500"
                  : !input.trim()
                    ? "text-gray-400 cursor-not-allowed"
                    : "text-white hover:opacity-80"
                  }`}
                style={{ lineHeight: "32px" }}
              >
                {isLoading ? t("chat.stop") : t("chat.send")}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
