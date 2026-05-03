import { useState, useRef, useCallback, type KeyboardEvent } from 'react';
import { Send, Square } from 'lucide-react';
import { useSettingsStore } from '@/stores/settingsStore';
import { useCaptainStore } from '@/stores/captainStore';

interface ChatInputProps {
  onSend: (content: string) => void;
  onStop: () => void;
}

export default function ChatInput({ onSend, onStop }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isProcessing = useCaptainStore((s) => s.isProcessing);
  const { provider, model } = useSettingsStore((s) => s.llmConfig);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isProcessing) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [value, isProcessing, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const maxHeight = 6 * 24;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, []);

  return (
    <div className="border-t border-aware-border bg-aware-panel px-4 py-3">
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            handleInput();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Message Captain..."
          disabled={isProcessing}
          rows={1}
          className="flex-1 resize-none rounded-lg border border-aware-border bg-aware-surface px-3 py-2.5 text-sm text-aware-text placeholder:text-aware-muted focus:border-aware-captain focus:outline-none disabled:opacity-50"
        />
        {isProcessing ? (
          <button
            onClick={onStop}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-aware-error/20 text-aware-error hover:bg-aware-error/30 transition-colors"
            title="Stop"
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!value.trim()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-aware-captain/20 text-aware-captain hover:bg-aware-captain/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Send"
          >
            <Send size={16} />
          </button>
        )}
      </div>
      <p className="mt-1.5 text-[10px] text-aware-muted">
        {provider}/{model || 'no model selected'} &middot; Enter to send, Shift+Enter for newline
      </p>
    </div>
  );
}
