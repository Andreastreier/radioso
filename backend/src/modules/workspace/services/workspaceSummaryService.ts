import type { ConversationRepositoryPort } from "../../../db/repositories/conversationRepository.js";
import type { ChatReadiness } from "../../../shared/infra/llm/chatReadiness.js";
import type { DocumentRepositoryPort } from "../../documents/contracts/index.js";

interface WorkspaceSummary {
  documentCount: number;
  readyDocumentCount: number;
  pendingDocumentCount: number;
  sampleDocumentCount: number;
  sampleDocumentSlugs: string[];
  conversationCount: number;
  hasDocuments: boolean;
  hasPendingDocuments: boolean;
  hasReadyDocuments: boolean;
  hasCompletedChat: boolean;
  sampleDocumentsImported: boolean;
  websiteCrawlerEnabled: boolean;
  /** Whether the workspace can make chat model calls, so callers can offer LLM features or a setup step. */
  llmConfigured: boolean;
}

export class WorkspaceSummaryService {
  constructor(
    private readonly documentRepository: DocumentRepositoryPort,
    private readonly conversationRepository: ConversationRepositoryPort,
    private readonly options: {
      websiteCrawlerEnabled: boolean;
      chatReadiness: Pick<ChatReadiness, "isChatConfigured">;
    },
  ) {}

  async getSummary(workspaceId: string): Promise<WorkspaceSummary> {
    const [documentSummary, conversationCount, llmConfigured] = await Promise.all([
      this.documentRepository.summarizeWorkspace(workspaceId),
      this.conversationRepository.countByWorkspaceId(workspaceId),
      this.options.chatReadiness.isChatConfigured(workspaceId),
    ]);

    // Projected field by field rather than spread: this summary is the onboarding
    // contract, so it must not grow silently when the document record does.
    return {
      documentCount: documentSummary.documentCount,
      readyDocumentCount: documentSummary.readyDocumentCount,
      pendingDocumentCount: documentSummary.pendingDocumentCount,
      sampleDocumentCount: documentSummary.sampleDocumentCount,
      sampleDocumentSlugs: documentSummary.sampleDocumentSlugs,
      conversationCount,
      hasDocuments: documentSummary.documentCount > 0,
      hasPendingDocuments: documentSummary.pendingDocumentCount > 0,
      hasReadyDocuments: documentSummary.readyDocumentCount > 0,
      hasCompletedChat: conversationCount > 0,
      sampleDocumentsImported: documentSummary.sampleDocumentCount > 0,
      websiteCrawlerEnabled: this.options.websiteCrawlerEnabled,
      llmConfigured,
    };
  }
}
