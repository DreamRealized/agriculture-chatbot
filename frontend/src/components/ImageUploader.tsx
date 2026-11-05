import { useState, useCallback, useRef, useEffect } from "react";
import { ImagePlus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

type ImageUploaderProps = {
  onImagesChange: (images: string[]) => void;
  disabled?: boolean;
  resetTrigger?: boolean;
};

export const ImageUploader = ({
  onImagesChange,
  disabled = false,
  resetTrigger = false,
}: ImageUploaderProps) => {
  const { t } = useTranslation();
  const [images, setImages] = useState<string[]>([]);
  const [readError, setReadError] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (resetTrigger) {
      setImages([]);
      setReadError(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [resetTrigger]);

  const handleImageUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (!files || disabled) return;
      setReadError(false);
      try {
        const newImages = await Promise.all(Array.from(files).map(file => (
          new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => typeof reader.result === "string"
              ? resolve(reader.result) : reject(new Error("Invalid image data"));
            reader.onerror = () => reject(reader.error);
            reader.onabort = () => reject(new Error("Image read aborted"));
            reader.readAsDataURL(file);
          })
        )));
        const updatedImages = [...images, ...newImages];
        setImages(updatedImages);
        onImagesChange(updatedImages);
        if (fileInputRef.current) fileInputRef.current.value = "";
      } catch {
        setReadError(true);
      }
    },
    [disabled, images, onImagesChange],
  );

  const removeImage = (index: number) => {
    const newImages = images.filter((_, i) => i !== index);
    setImages(newImages);
    onImagesChange(newImages);
  };

  return (
    <div className="chat-upload-control">
      {images.map((image, index) => (
        <div key={index} className="chat-upload-preview">
          <img src={image} alt={t("chat.attachment", { number: index + 1 })} />
          <Tooltip>
            <TooltipTrigger asChild>
              <button type="button" className="chat-remove-image" disabled={disabled}
                aria-label={t("chat.removeImage", { number: index + 1 })}
                onClick={() => removeImage(index)}>
                <X size={12} aria-hidden="true" />
              </button>
            </TooltipTrigger>
            <TooltipContent>{t("chat.removeImage", { number: index + 1 })}</TooltipContent>
          </Tooltip>
        </div>
      ))}
      <Tooltip>
        <TooltipTrigger asChild>
          <button type="button" className="chat-icon-button" disabled={disabled}
            aria-label={t("chat.file_uploader_title")}
            onClick={() => fileInputRef.current?.click()}>
            <ImagePlus size={19} strokeWidth={1.7} aria-hidden="true" />
          </button>
        </TooltipTrigger>
        <TooltipContent>{t("chat.file_uploader_title")}</TooltipContent>
      </Tooltip>
      <input ref={fileInputRef} type="file" accept="image/*" multiple
        onChange={handleImageUpload} disabled={disabled} hidden />
      {readError && <span className="chat-control-error" role="alert">{t("chat.uploadError")}</span>}
    </div>
  );
};
