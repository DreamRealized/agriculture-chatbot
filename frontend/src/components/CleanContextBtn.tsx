import { useCallback, useState } from "react";
import { Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { sendCleanContextRequest } from "@/lib/ollamaService";

type CleanContextBtnProps = {
  cleanContext: () => void;
  chatId: string;
  disabled?: boolean;
};

export const CleanContextBtn = ({ cleanContext, chatId, disabled = false }: CleanContextBtnProps) => {
  const { t } = useTranslation();
  const [isClearing, setIsClearing] = useState(false);
  const [hasError, setHasError] = useState(false);
  const onCleanContext = useCallback(async () => {
    setIsClearing(true);
    setHasError(false);
    try {
      await sendCleanContextRequest(chatId);
      cleanContext();
    } catch {
      setHasError(true);
    } finally {
      setIsClearing(false);
    }
  }, [chatId, cleanContext]);

  return (
    <div className="chat-clear-control">
      <Tooltip>
        <TooltipTrigger asChild>
          <button type="button" className="chat-icon-button" onClick={onCleanContext}
            disabled={disabled || isClearing} aria-label={t("chat.clearContext")} aria-busy={isClearing}>
            <Trash2 size={18} strokeWidth={1.7} aria-hidden="true" />
          </button>
        </TooltipTrigger>
        <TooltipContent>{t(isClearing ? "chat.clearing" : "chat.clearContext")}</TooltipContent>
      </Tooltip>
      {hasError && <span className="chat-control-error" role="alert">{t("chat.clearError")}</span>}
    </div>
  );
};
