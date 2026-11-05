import React, { useState, useRef, useEffect, useMemo } from "react";
import type { Message } from "@/types/types";
import { generateResponse, sendCleanContextRequest } from "@/lib/ollamaService";
import { ImageUploader } from "./ImageUploader";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { useTranslation } from "react-i18next";
import { ArrowUp, MessageSquare, Plus, Square } from "lucide-react";
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
  const { t, i18n } = useTranslation();
  const isChinese = i18n.language.startsWith("zh");
  const copy = {
    newChat: t("chat.newChat"), heading: t("chat.welcome"), subtitle: t("chat.welcomeSubtitle"),
    language: t("chat.language"), you: t("chat.you"), assistant: t("chat.assistant"),
    clearing: t("chat.clearing"), hint: t("chat.composerHint"), clearError: t("chat.clearError"),
  };
  const [isResetting, setIsResetting] = useState(false);
  const focusAfterResetRef = useRef(false);
  const [resetError, setResetError] = useState("");
  const savedMessagesRef = useRef<Set<string>>(new Set());
  const [targetDB, setTargetDB] = useState("");
  const conversationId = useMemo(() => `chat_${Date.now()}`, [])
  const { addToAudioQueue, flushBuffer, clearQueue } = useAudioPlayer()

  // Restore focus after React has re-enabled the composer following New Chat.
  useEffect(() => {
    if (!isResetting && focusAfterResetRef.current) {
      focusAfterResetRef.current = false;
      inputRef.current?.focus();
    }
  }, [isResetting]);

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
        const error = new Error(t("chat.saveMessageError"));
        logError(t("chat.saveMessageError"), error, {
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
              text: t("chat.error", { message: error.message }),
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
          text: t("chat.sendError", { message: error instanceof Error ? error.message : t("chat.unknownError") }),
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
          text: t("chat.stopped"),
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

  const MemoizedMessage = React.memo(({ msg }: { msg: Message }) => (
    <article className={`chat-message chat-message--${msg.sender}`} aria-label={msg.sender === "user" ? copy.you : copy.assistant}>
      <div className="chat-message-body">
        {msg.images && msg.images.length > 0 && (
          <div className="chat-message-images">
            {msg.images.map((image, index) => (
              <img key={index} src={image} alt={t("chat.attachment", { number: index + 1 })} />
            ))}
          </div>
        )}
        <MarkdownRenderer content={msg.text} />
      </div>
    </article>
  ));

  const cleanContext = React.useCallback(() => {
    setMessages([])
  }, [])

  const startNewChat = async () => {
    if (isLoading || isResetting) return;
    setIsResetting(true);
    setResetError("");
    try {
      await sendCleanContextRequest(conversationId);
      cleanContext();
      setInput("");
      setShouldResetImages(true);
      setTargetDB("");
      clearQueue();
      focusAfterResetRef.current = true;
    } catch {
      setResetError(copy.clearError);
    } finally {
      setIsResetting(false);
    }
  };

  const newChatButton = (
    <button type="button" className="chat-new-button" onClick={startNewChat} disabled={isLoading || isResetting}>
      <Plus size={18} aria-hidden="true" />
      <span>{isResetting ? copy.clearing : copy.newChat}</span>
    </button>
  );

  return (
    <div className="chat-app">
      <aside className="chat-sidebar" aria-label="AI Chat">
        <div className="chat-brand"><MessageSquare size={21} strokeWidth={1.6} aria-hidden="true" /><span>AI Chat</span></div>
        {newChatButton}
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <span className="chat-header-title">AI Chat</span>
          <div className="chat-mobile-new">{newChatButton}</div>
          <select
            className="chat-language"
            aria-label={copy.language}
            value={isChinese ? "zh" : "en"}
            onChange={event => { void i18n.changeLanguage(event.target.value); }}
          >
            <option value="en">English</option>
            <option value="zh">中文</option>
          </select>
        </header>

        <div className="chat-scroll">
          <div className={`chat-transcript ${messages.length === 0 ? "chat-transcript--empty" : ""}`}>
            {messages.length === 0 && (
              <div className="chat-empty">
                <h1>{copy.heading}</h1>
                <p>{copy.subtitle}</p>
              </div>
            )}
            {messages.map((msg, index) => <MemoizedMessage key={index} msg={msg} />)}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="chat-composer-region">
          {resetError && <p className="chat-error" role="alert">{resetError}</p>}
          <div className="chat-composer">
            <textarea
              ref={inputRef}
              value={input}
              onChange={event => setInput(event.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t(images.length > 0 ? "chat.file_placeholder" : "chat.placeholder")}
              aria-label={t(images.length > 0 ? "chat.file_placeholder" : "chat.placeholder")}
              disabled={isLoading || isResetting}
              rows={2}
            />
            <div className="chat-composer-toolbar">
              <div className="chat-composer-tools">
                <ImageUploader
                  onImagesChange={imgs => setImages(imgs)}
                  disabled={isLoading || isResetting}
                  resetTrigger={shouldResetImages}
                />
                <div className="chat-tool">
                  <SpeechToText
                    disabled={isLoading || isResetting}
                    afterTranslate={str => setInput(prev => prev + str)}
                    onVoiceInActive={() => {
                      const timeout = setTimeout(() => {
                        sendMessage();
                        clearTimeout(timeout);
                      }, 500);
                    }}
                  />
                </div>
                <div className="chat-tool" inert={isLoading || isResetting}>
                  <CleanContextBtn cleanContext={cleanContext} chatId={conversationId} disabled={isLoading || isResetting} />
                </div>
              </div>
              <button
                type="button"
                className="chat-send-button"
                onClick={isLoading ? stopGeneration : sendMessage}
                disabled={isResetting || (!isLoading && !input.trim())}
                aria-label={isLoading ? t("chat.stop") : t("chat.send")}
                title={isLoading ? t("chat.stop") : t("chat.send")}
              >
                {isLoading ? <Square size={15} fill="currentColor" aria-hidden="true" /> : <ArrowUp size={21} aria-hidden="true" />}
              </button>
            </div>
          </div>
          <p className="chat-composer-hint">{copy.hint}</p>
        </div>
      </main>
    </div>
  );
};
