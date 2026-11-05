/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef, useState } from "react";
import pinyin from "pinyin";
import { useTranslation } from "react-i18next";

interface VoiceWakeupProps {
  onWakeup: () => void;
  wakeupKeyword?: string;
}

export const VoiceWakeup: React.FC<VoiceWakeupProps> = ({
  onWakeup,
  wakeupKeyword = "你好助手"
}) => {
  const { t, i18n } = useTranslation();
  const [isListening, setIsListening] = useState(false);
  const [isPaused, setIsPaused] = useState(false); // 新增：用于控制是否暂停唤醒检测
  const recognitionRef = useRef<any>(null);
  const restartTimeoutRef = useRef<any>(null);
  const hasProcessedRef = useRef(false); // 新增：用于防止重复处理同一次唤醒

  // 将文字转换为拼音（不带声调）
  const convertToPinyin = (text: string): string => {
    return pinyin(text, {
      style: pinyin.STYLE_NORMAL, // 不带声调
      segment: true // 启用分词
    })
      .flat()
      .join("");
  };

  // 检查两段文字的拼音是否匹配
  const isPinyinMatch = (text1: string, text2: string): boolean => {
    const pinyin1 = convertToPinyin(text1)
      .toLowerCase()
      .replace(/[^a-z]/g, "");
    const pinyin2 = convertToPinyin(text2)
      .toLowerCase()
      .replace(/[^a-z]/g, "");

    return pinyin1.includes(pinyin2) || pinyin2.includes(pinyin1);
  };

  // 新增：重置处理状态的函数
  const resetProcessingState = () => {
    hasProcessedRef.current = false;
  };

  // 新增：恢复语音唤醒检测的函数
  const resumeWakeupDetection = () => {
    setIsPaused(false);
    resetProcessingState();
    startListening();
  };

  // 新增：暂停语音唤醒检测的函数
  const pauseWakeupDetection = () => {
    setIsPaused(true);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error("您的浏览器不支持语音识别功能");
      return;
    }

    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;

    recognition.continuous = false;
    recognition.interimResults = false; // 改为 false，只获取最终结果
    recognition.lang = "zh-CN";
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      console.log("语音识别已启动");
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      // 如果当前是暂停状态或已经处理过，直接返回
      if (isPaused || hasProcessedRef.current) return;

      const results: any[] = Array.from(event.results);
      const lastResult = results[results.length - 1];
      const text = lastResult[0].transcript;

      if (isPinyinMatch(text, wakeupKeyword)) {
        console.log("检测到唤醒词匹配！原文:", text, "唤醒词:", wakeupKeyword);

        // 标记已处理并暂停检测
        hasProcessedRef.current = true;
        setIsPaused(true);

        // 停止当前识别
        recognition.stop();

        if ("speechSynthesis" in window) {
          const speech = new SpeechSynthesisUtterance(t("chat.voiceReady"));
          speech.lang = i18n.language.startsWith("zh") ? "zh-CN" : "en-US";
          speech.rate = 1;
          speech.pitch = 1;
          speech.volume = 1;

          // 设置一个超时保护，确保即使语音合成没有正常工作，也会继续执行
          const timeoutId = setTimeout(() => {
            console.log("语音合成超时，继续执行");
            window.speechSynthesis.cancel(); // 取消可能的待处理语音
            onWakeup();
          }, 3000); // 3秒后如果还没有播放完成，就强制继续

          // 添加更多事件监听器来帮助调试
          speech.onstart = () => {
            console.log("语音合成开始播放");
          };

          speech.onend = () => {
            console.log("语音合成播放结束");
            clearTimeout(timeoutId); // 清除超时保护
            onWakeup();
          };

          speech.onerror = event => {
            console.error("语音合成错误:", event);
            clearTimeout(timeoutId); // 清除超时保护
            onWakeup();
          };

          speech.onpause = () => {
            console.log("语音合成被暂停");
          };

          speech.onresume = () => {
            console.log("语音合成已恢复");
          };

          // 取消所有正在进行的语音
          window.speechSynthesis.cancel();

          // 确保语音合成服务处于正确状态
          window.speechSynthesis.cancel(); // 先取消所有待处理的语音

          // 使用 setTimeout 确保语音列表已加载
          setTimeout(() => {
            try {
              window.speechSynthesis.speak(speech);
              console.log("语音合成speak方法已调用");
            } catch (innerError) {
              console.error("语音播放执行错误:", innerError);
              clearTimeout(timeoutId);
              onWakeup();
            }
          }, 100);
        } else {
          console.warn("浏览器不支持语音合成");
          onWakeup();
        }
      }
    };

    recognition.onend = () => {
      console.log("语音识别已结束");
      setIsListening(false);

      // 清除之前的定时器
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }

      // 只有在非暂停状态且未处理唤醒词的情况下才重新启动
      if (!isPaused && !hasProcessedRef.current) {
        restartTimeoutRef.current = setTimeout(() => {
          if (!recognitionRef.current || isPaused) return;
          try {
            recognitionRef.current.start();
            console.log("重新启动语音识别");
          } catch (error) {
            console.error("重启语音识别失败:", error);
          }
        }, 500);
      }
    };

    recognition.onerror = (event: any) => {
      console.error("语音识别错误:", event.error);
      setIsListening(false);

      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }

      // 只有在非暂停状态、未处理唤醒词且错误不是 not-allowed 时才重新启动
      if (
        event.error !== "not-allowed" &&
        !isPaused &&
        !hasProcessedRef.current
      ) {
        restartTimeoutRef.current = setTimeout(() => {
          startListening();
        }, 1000);
      }
    };

    // 只有在非暂停状态下才初始启动
    if (!isPaused) {
      startListening();
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }
    };
  }, [onWakeup, wakeupKeyword, isPaused, t, i18n.language]); // 添加 isPaused 作为依赖

  const startListening = () => {
    try {
      if (!recognitionRef.current || isListening || isPaused) return;

      recognitionRef.current.start();
      console.log("开始语音识别");
    } catch (error) {
      console.error("启动语音识别失败:", error);
      setIsListening(false);
    }
  };

  // 暴露恢复检测的方法给父组件
  useEffect(() => {
    // 将恢复函数添加到 window 对象上，方便外部调用
    (window as any).resumeWakeupDetection = resumeWakeupDetection;
    (window as any).pauseWakeupDetection = pauseWakeupDetection; // 暴露暂停函数

    return () => {
      delete (window as any).resumeWakeupDetection;
      delete (window as any).pauseWakeupDetection; // 清理暂停函数
    };
  }, []);

  return null;
};
