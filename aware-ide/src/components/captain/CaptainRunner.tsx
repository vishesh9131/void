import { useCallback, useRef } from 'react';
import { nanoid } from 'nanoid';
import { useCaptainStore } from '@/stores/captainStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useWorkerStore } from '@/stores/workerStore';
import { useKanbanStore } from '@/stores/kanbanStore';
import { runCaptainTurn } from '@/services/agents/captain';
import type { CaptainContext } from '@/services/agents/captain';
import type { CanvasAction, KanbanAction } from '@/services/agents/tools';
import type { ChatMessage } from '@/types/agents';
import type { TicketStatus } from '@/types/kanban';

export function useCaptainRunner() {
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    const captainState = useCaptainStore.getState();
    const { activeSessionId, sessions, setProcessing, addMessage } = captainState;

    if (!activeSessionId) return;
    const session = sessions.find((s) => s.id === activeSessionId);
    if (!session) return;

    const config = useSettingsStore.getState().llmConfig;
    const workers = Object.values(useWorkerStore.getState().workers);
    const kanbanState = useKanbanStore.getState();
    const tickets = Object.values(kanbanState.tickets);
    const projectPath = useSettingsStore.getState().projectPath ?? '.';

    const onCanvasAction = (_action: CanvasAction) => {
      // Canvas integration handled by parent layout
    };

    const onKanbanAction = (action: KanbanAction) => {
      const store = useKanbanStore.getState();
      if (action.type === 'create_ticket') {
        store.addTicket({
          ...action.ticket,
          reporterId: 'captain',
          tags: [],
          subtasks: [],
          comments: [],
          linkedNodeIds: [],
        });
      } else if (action.type === 'update_ticket') {
        if (action.status) {
          store.moveTicket(action.ticketId, action.status as TicketStatus);
        }
        if (action.comment) {
          store.addComment(action.ticketId, { author: 'Captain', content: action.comment });
        }
      }
    };

    const context: CaptainContext = {
      projectPath,
      onCanvasAction,
      onKanbanAction,
      workers,
      kanbanTickets: tickets,
    };

    const sessionCopy = { ...session, messages: [...session.messages] };
    const abortController = new AbortController();
    abortRef.current = abortController;
    setProcessing(true);

    let assistantText = '';
    let assistantThinking = '';
    const toolCalls: ChatMessage['toolCalls'] = [];

    try {
      const gen = runCaptainTurn(sessionCopy, config, context);

      for await (const event of gen) {
        if (abortController.signal.aborted) break;

        switch (event.type) {
          case 'text':
            assistantText += event.content;
            break;
          case 'thinking':
            assistantThinking += event.content;
            break;
          case 'tool_call':
            if (event.call) {
              toolCalls.push(event.call);
            }
            break;
          case 'tool_result':
            break;
          case 'done': {
            if (assistantText || toolCalls.length > 0) {
              const msg: ChatMessage = {
                id: nanoid(),
                role: 'assistant',
                content: assistantText,
                timestamp: Date.now(),
                thinking: assistantThinking || undefined,
                toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined,
              };
              addMessage(activeSessionId, msg);
            }
            break;
          }
          case 'error': {
            const errMsg: ChatMessage = {
              id: nanoid(),
              role: 'assistant',
              content: `Error: ${event.message}`,
              timestamp: Date.now(),
            };
            addMessage(activeSessionId, errMsg);
            break;
          }
        }
      }
    } catch (err) {
      const errorContent = err instanceof Error ? err.message : String(err);
      const errMsg: ChatMessage = {
        id: nanoid(),
        role: 'assistant',
        content: `Error: ${errorContent}`,
        timestamp: Date.now(),
      };
      addMessage(activeSessionId, errMsg);
    } finally {
      setProcessing(false);
      abortRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    useCaptainStore.getState().setProcessing(false);
  }, []);

  return { run, stop };
}
