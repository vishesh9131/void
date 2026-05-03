import { useState, useCallback } from 'react';
import { RefreshCw } from 'lucide-react';
import { useSettingsStore } from '@/stores/settingsStore';
import { createLLMService } from '@/services/llm';
import type { LLMProvider } from '@/types/llm';
import { Input } from '@/components/shared/Input';
import Button from '@/components/shared/Button';
import ProviderCard from './ProviderCard';

type ConnectionStatus = 'connected' | 'error' | 'unconfigured';

const PROVIDERS: LLMProvider[] = ['anthropic', 'openai', 'vllm'];

function getConnectionStatus(provider: LLMProvider, apiKey?: string, baseUrl?: string): ConnectionStatus {
  if (provider === 'vllm') {
    return baseUrl ? 'connected' : 'unconfigured';
  }
  return apiKey ? 'connected' : 'unconfigured';
}

export default function SettingsPanel() {
  const llmConfig = useSettingsStore((s) => s.llmConfig);
  const availableModels = useSettingsStore((s) => s.availableModels);
  const setProvider = useSettingsStore((s) => s.setProvider);
  const setModel = useSettingsStore((s) => s.setModel);
  const setApiKey = useSettingsStore((s) => s.setApiKey);
  const setBaseUrl = useSettingsStore((s) => s.setBaseUrl);
  const setTemperature = useSettingsStore((s) => s.setTemperature);
  const setMaxTokens = useSettingsStore((s) => s.setMaxTokens);
  const setAvailableModels = useSettingsStore((s) => s.setAvailableModels);

  const [testStatus, setTestStatus] = useState<string | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [fetchingModels, setFetchingModels] = useState(false);

  const handleTestConnection = useCallback(async () => {
    setTestLoading(true);
    setTestStatus(null);
    try {
      const service = createLLMService(llmConfig);
      const models = await service.listModels();
      setAvailableModels(models);
      setTestStatus(`OK -- ${models.length} model(s) found`);
    } catch (err) {
      setTestStatus(`Failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTestLoading(false);
    }
  }, [llmConfig, setAvailableModels]);

  const handleFetchModels = useCallback(async () => {
    setFetchingModels(true);
    try {
      const service = createLLMService(llmConfig);
      const models = await service.listModels();
      setAvailableModels(models);
    } catch {
      // silently fail -- user can use Test Connection for diagnostics
    } finally {
      setFetchingModels(false);
    }
  }, [llmConfig, setAvailableModels]);

  const renderProviderFields = (provider: LLMProvider) => {
    const isActive = llmConfig.provider === provider;
    if (!isActive) return null;

    return (
      <div className="flex flex-col gap-4">
        {provider !== 'vllm' ? (
          <Input
            label="API Key"
            type="password"
            placeholder={`Enter ${provider === 'anthropic' ? 'Anthropic' : 'OpenAI'} API key`}
            value={llmConfig.apiKey ?? ''}
            onChange={(e) => setApiKey(e.target.value)}
          />
        ) : (
          <Input
            label="Base URL"
            type="text"
            placeholder="https://vllm.corerec.online/v1"
            value={llmConfig.baseUrl ?? ''}
            onChange={(e) => setBaseUrl(e.target.value)}
          />
        )}

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-aware-muted">Model</label>
          {availableModels.length > 0 ? (
            <select
              className="w-full rounded-lg border border-aware-border bg-aware-bg px-3 py-2 text-sm text-aware-text focus:border-aware-accent focus:outline-none focus:ring-1 focus:ring-aware-accent/40"
              value={llmConfig.model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="">Select a model</option>
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <div className="flex gap-2">
              <input
                className="flex-1 rounded-lg border border-aware-border bg-aware-bg px-3 py-2 text-sm text-aware-text placeholder:text-aware-muted/60 focus:border-aware-accent focus:outline-none focus:ring-1 focus:ring-aware-accent/40"
                placeholder="e.g. claude-sonnet-4-20250514"
                value={llmConfig.model}
                onChange={(e) => setModel(e.target.value)}
              />
              {provider === 'vllm' && (
                <Button
                  variant="ghost"
                  size="sm"
                  loading={fetchingModels}
                  icon={<RefreshCw size={13} />}
                  onClick={handleFetchModels}
                  title="Fetch models"
                >
                  Fetch
                </Button>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-aware-muted">
              Temperature
            </label>
            <span className="text-xs tabular-nums text-aware-text">
              {llmConfig.temperature.toFixed(1)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={llmConfig.temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full accent-aware-accent"
          />
        </div>

        <Input
          label="Max Tokens"
          type="number"
          min={1}
          max={200000}
          value={llmConfig.maxTokens}
          onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 4096)}
        />

        <div className="flex items-center gap-2 pt-2">
          <Button
            variant="primary"
            size="sm"
            loading={testLoading}
            onClick={handleTestConnection}
          >
            Test Connection
          </Button>

          {testStatus && (
            <span
              className={`text-xs ${
                testStatus.startsWith('OK') ? 'text-aware-success' : 'text-aware-error'
              }`}
            >
              {testStatus}
            </span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-full flex-col bg-aware-bg">
      <div className="border-b border-aware-border bg-aware-panel px-4 py-2.5">
        <span className="text-xs font-semibold uppercase tracking-wider text-aware-muted">
          Settings
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-md space-y-3">
          <h3 className="mb-2 text-sm font-medium text-aware-text">
            LLM Provider
          </h3>
          {PROVIDERS.map((p) => (
            <ProviderCard
              key={p}
              provider={p}
              isActive={llmConfig.provider === p}
              status={getConnectionStatus(p, llmConfig.apiKey, llmConfig.baseUrl)}
              onClick={() => setProvider(p)}
            >
              {renderProviderFields(p)}
            </ProviderCard>
          ))}
        </div>
      </div>
    </div>
  );
}
