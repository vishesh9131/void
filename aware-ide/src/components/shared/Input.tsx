import { type InputHTMLAttributes, type TextareaHTMLAttributes, forwardRef } from 'react';

interface InputBaseProps {
  label?: string;
  error?: string;
}

type InputProps = InputBaseProps & InputHTMLAttributes<HTMLInputElement>;
type TextareaProps = InputBaseProps & TextareaHTMLAttributes<HTMLTextAreaElement>;

const baseClasses =
  'w-full rounded-lg border border-aware-border bg-aware-bg px-3 py-2 text-sm text-aware-text placeholder:text-aware-muted/60 focus:border-aware-accent focus:outline-none focus:ring-1 focus:ring-aware-accent/40 transition-colors';

const errorClasses =
  'border-aware-error focus:border-aware-error focus:ring-aware-error/40';

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className = '', ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-xs font-medium text-aware-muted">{label}</label>
      )}
      <input
        ref={ref}
        className={`${baseClasses} ${error ? errorClasses : ''} ${className}`}
        {...props}
      />
      {error && <span className="text-xs text-aware-error">{error}</span>}
    </div>
  ),
);
Input.displayName = 'Input';

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, error, className = '', ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-xs font-medium text-aware-muted">{label}</label>
      )}
      <textarea
        ref={ref}
        className={`${baseClasses} resize-y ${error ? errorClasses : ''} ${className}`}
        rows={4}
        {...props}
      />
      {error && <span className="text-xs text-aware-error">{error}</span>}
    </div>
  ),
);
Textarea.displayName = 'Textarea';
