import { createHash } from "node:crypto";

import { GENERATION_SURFACE } from "../../../shared/domain/generationSurface.js";
import { canonicalJson } from "../../../shared/domain/canonicalJson.js";
import { addressesSurface, effectiveSurfaces } from "../../../shared/domain/steeringRule.js";
import type {
  Directive,
  DirectiveCoherenceCheckInput,
  DirectiveCoherenceChecker,
  DirectiveCoherenceVerdict,
} from "@radioso/conversation-contract";

import type { AgentDirectiveUpdateOptions, AgentRepositoryPort } from "../../../db/repositories/agentRepository.js";
import { badRequest, conflict, notFound } from "../../../shared/domain/errors.js";
import type { AgentSkillRepositoryPort } from "../../agentSkills/public.js";
import { defaultAnswerDirectives } from "../../directives/public.js";
import {
  authoredDirectiveInputSchema,
  validateAuthoredDirectiveCapabilities,
  type AuthoredDirective,
  type AuthoredDirectiveInput,
  type NormalizedAuthoredDirectiveInput,
} from "../authoredDirectives.js";
import type { AnswerCoverageCriteria } from "../../answerCoverage/public.js";
import { authoredDirectiveToDirective } from "../authoredDirectiveMapper.js";
import type { AgentRecord } from "../domain.js";

interface AuthoredDirectiveSaveResult {
  directive: AuthoredDirective;
  coherence: DirectiveCoherenceVerdict;
}

type AuthoredDirectiveVersionOptions = AgentDirectiveUpdateOptions;

// Every key the input schema declares, read from the schema itself rather than hand-listed, so a
// field added to `authoredDirectiveInputSchema` is carried forward on update automatically instead
// of silently resetting to its schema default the next time someone edits an unrelated field.
const authoredDirectiveInputKeys = Object.keys(authoredDirectiveInputSchema.shape) as Array<keyof AuthoredDirectiveInput>;

/**
 * Carries every schema field forward from the stored directive unless the caller's patch names it
 * explicitly - `hasOwnProperty`, not `??`, is what tells "omitted" apart from "explicitly cleared
 * with `null` or `[]`". A PATCH body only ever contains the keys the operator actually touched, so
 * an absent key must mean "leave this alone." This matters most for a field the schema defaults to
 * empty on absence, such as `surfaces`: omitting it keeps the directive's stored scope, while an
 * explicit `[]` is the operator widening it back to the answering voice. Treating an omitted key as
 * `??`'s "value ?? default" would silently narrow or reset a field like that on every unrelated
 * edit.
 */
type AuthoredDirectivePatchInput = Omit<Partial<AuthoredDirectiveInput>, "coverageCriteria"> & {
  coverageCriteria?: AnswerCoverageCriteria | null;
};

const carryForwardAuthoredDirectiveInput = (
  input: AuthoredDirectivePatchInput,
  existing: AuthoredDirective,
): AuthoredDirectiveInput =>
  Object.fromEntries(
    authoredDirectiveInputKeys.map((key) => [
      key,
      key === "coverageCriteria" && input.coverageCriteria === null
        ? undefined
        : Object.prototype.hasOwnProperty.call(input, key) ? input[key] : existing[key],
    ]),
    // `existing` is a prior successful parse of this same schema, so every value read from it here
    // is already valid raw input for the field it came from; the cast just restates that as the
    // input type Object.fromEntries' generic widens away.
  ) as AuthoredDirectiveInput;

type AuthoredDirectiveAgentContext = Pick<
  AgentRecord,
  "id" | "name" | "customInstruction" | "greetingInstruction" | "assistantDefaultLocale" | "chatModelOverride"
>;

export interface AuthoredDirectiveServiceOptions {
  repository: Pick<AgentRepositoryPort, "listDirectives" | "createDirective" | "updateDirective" | "deleteDirective"> & {
    findByIdAndWorkspaceId(agentId: string, workspaceId: string): Promise<AuthoredDirectiveAgentContext | null>;
  };
  coherenceChecker: DirectiveCoherenceChecker;
  registeredCapabilityNames: ReadonlySet<string>;
  agentSkills?: Pick<AgentSkillRepositoryPort, "findByName">;
}

const coherenceUnavailableVerdict = (): DirectiveCoherenceVerdict => ({
  coherent: true,
  conflicts: [],
  rationale: "Coherence check unavailable.",
});

/** One directive to check before it is saved; `ref` stays stable until it has a real id. */
export interface DirectiveCoherenceCandidate {
  ref: string;
  input: AuthoredDirectiveInput;
}

export interface DirectiveCoherenceAssessment {
  verdicts: Array<{ ref: string; verdict: DirectiveCoherenceVerdict }>;
  /** SHA-256 over every input the checker read, as built by `buildCoherenceCheckInput`. */
  contextToken: string;
}

const PENDING_DIRECTIVE_ID_PREFIX = "pending-directive:";

/** How a not-yet-saved candidate is referenced in coherence inputs and conflicts. */
export const pendingDirectiveId = (ref: string): string => `${PENDING_DIRECTIVE_ID_PREFIX}${ref}`;

/** What a candidate is compared with: enabled directives, then earlier candidates, then built-ins. */
const comparisonPool = (existingDirectives: AuthoredDirective[], earlierCandidates: Directive[] = []): Directive[] => [
  // A disabled directive cannot fire, so it cannot conflict with the candidate;
  // including it would flag noise the operator can't act on.
  ...existingDirectives.filter((directive) => directive.enabled).map((directive) => authoredDirectiveToDirective(directive)),
  ...earlierCandidates,
  ...defaultAnswerDirectives,
];

/**
 * The one place a coherence check's input is built. The checker receives exactly this, and the
 * context token hashes exactly this, so every model input (agent instructions, locale and model
 * override, the candidate, the comparison set narrowed to the candidate's generation surfaces,
 * and the built-in defaults) is covered by the token.
 *
 * The model that judges coherence is not part of the input because it cannot vary per workspace:
 * the chat pipeline the checker runs on pins the deployment's chat provider and model, and the
 * checker's prompt is a constant shipped with `@radioso/conversation-defaults`. If either becomes
 * workspace-configurable, add it here. `directive-coherence-inference-config.test.ts` fails first.
 */
const buildCoherenceCheckInput = (
  workspaceId: string,
  agent: AuthoredDirectiveAgentContext,
  candidate: Directive,
  pool: Directive[],
): DirectiveCoherenceCheckInput => ({
  invocationContext: { workspaceId, agentId: agent.id },
  agent: {
    id: agent.id,
    name: agent.name,
    instructions: [agent.customInstruction, agent.greetingInstruction].filter((instruction) => instruction.trim().length > 0),
    defaultLocale: agent.assistantDefaultLocale,
    model: agent.chatModelOverride,
  },
  candidate,
  existingDirectives: pool.filter((directive) =>
    effectiveSurfaces(candidate.surfaces).some((surface) => addressesSurface(directive.surfaces, surface)),
  ),
});

const coherenceContextTokenOf = (inputs: ReadonlyArray<{ input: DirectiveCoherenceCheckInput }>): string =>
  createHash("sha256").update(canonicalJson(inputs.map((entry) => entry.input))).digest("hex");

const disabledCandidateVerdict = (): DirectiveCoherenceVerdict => ({
  coherent: true,
  conflicts: [],
  rationale: "Directive is disabled and cannot fire, so it cannot conflict with other directives; coherence was not checked.",
});

export class AuthoredDirectiveService {
  constructor(private readonly options: AuthoredDirectiveServiceOptions) {}

  async list(workspaceId: string, agentId: string): Promise<AuthoredDirective[]> {
    await this.requireAgent(workspaceId, agentId);
    return this.options.repository.listDirectives(agentId, workspaceId);
  }

  /**
   * `options.assessedCoherence` is a verdict `assessCoherence` produced for this exact input,
   * and is used instead of checking again. Pass it only after confirming, with
   * `coherenceContextToken`, that nothing the check read has changed since.
   */
  async create(
    workspaceId: string,
    agentId: string,
    input: AuthoredDirectiveInput,
    options: AuthoredDirectiveVersionOptions & { assessedCoherence?: DirectiveCoherenceVerdict } = {},
  ): Promise<AuthoredDirectiveSaveResult> {
    const { assessedCoherence, ...versionOptions } = options;
    const agent = await this.requireAgent(workspaceId, agentId);
    const directive = this.validateInput(input);
    await this.validateBinding(workspaceId, agentId, directive);
    const coherence = assessedCoherence
      ?? await this.checkCoherence(workspaceId, agent, directive, await this.options.repository.listDirectives(agentId, workspaceId));
    const saved = await this.options.repository.createDirective(agentId, workspaceId, {
      ...directive,
      routes: [],
    }, versionOptions);
    return { directive: saved, coherence };
  }

  async update(
    workspaceId: string,
    agentId: string,
    directiveId: string,
    input: AuthoredDirectivePatchInput,
    options?: AuthoredDirectiveVersionOptions,
  ): Promise<AuthoredDirectiveSaveResult> {
    const agent = await this.requireAgent(workspaceId, agentId);
    const existingDirectives = await this.options.repository.listDirectives(agentId, workspaceId);
    const existing = existingDirectives.find((directive) => directive.id === directiveId);
    if (!existing) {
      throw notFound("Directive not found");
    }
    const directive = this.validateInput(carryForwardAuthoredDirectiveInput(input, existing));
    await this.validateBinding(workspaceId, agentId, directive);
    const comparisonDirectives = existingDirectives.filter((directiveToCompare) => directiveToCompare.id !== directiveId);
    const coherence = await this.checkCoherence(workspaceId, agent, directive, comparisonDirectives);
    const saved = await this.options.repository.updateDirective(agentId, workspaceId, directiveId, {
      ...directive,
      routes: [],
    }, options);
    return { directive: saved, coherence };
  }

  /**
   * `options.expectedUpdatedAt`, when supplied, is enforced inside the repository's DELETE
   * predicate (not read-then-compared here): a concurrent edit changes the directive's
   * `updated_at` between when a caller read it and when it calls delete, and the WHERE clause
   * must see that change atomically or the delete could destroy a directive the caller never saw.
   */
  async delete(workspaceId: string, agentId: string, directiveId: string, options?: AuthoredDirectiveVersionOptions): Promise<void> {
    await this.requireAgent(workspaceId, agentId);
    const deleted = await this.options.repository.deleteDirective(agentId, workspaceId, directiveId, options);
    if (!deleted) {
      throw options?.expectedUpdatedAt ? conflict("Directive was updated by another writer; reload before saving again") : notFound("Directive not found");
    }
  }

  private validateInput(input: AuthoredDirectiveInput): NormalizedAuthoredDirectiveInput {
    const parsed = authoredDirectiveInputSchema.safeParse({
      ...input,
      routes: [],
    });
    if (!parsed.success) {
      throw badRequest("Invalid directive input", parsed.error.flatten());
    }
    const directive = parsed.data;
    // A disabled directive is out of play, so a capability that has since been removed or renamed
    // cannot reach a turn through it. Rejecting the save here would make the off switch unusable
    // in one of the cases it exists for, since an update carries the stored capabilities forward
    // untouched. Re-enabling validates in full, like the binding and coherence gates.
    if (directive.enabled) {
      const capabilityValidation = validateAuthoredDirectiveCapabilities(
        directive.requiredCapabilities,
        this.options.registeredCapabilityNames,
      );
      if (!capabilityValidation.ok) {
        throw badRequest("Directive references unknown capabilities", { unknown: capabilityValidation.unknown });
      }
    }
    return directive;
  }

  private async validateBinding(workspaceId: string, agentId: string, directive: NormalizedAuthoredDirectiveInput): Promise<void> {
    // A disabled directive is out of play and cannot fire, so a binding that would fail at
    // runtime is not rejected here - the authored text and its binding are preserved as
    // written, and the validation gate moves to the moment the directive comes back into
    // play. This is the same rule checkCoherence applies to disabled candidates, and it is
    // the point of the off switch: an operator must be able to disable a directive whose
    // binding broke (skill disabled or deleted) rather than being blocked from saving the
    // very action meant to stop it. Re-enabling runs this validation again in full.
    if (!directive.enabled) {
      return;
    }
    const binding = directive.binding;
    if (!binding) {
      return;
    }
    // Binding names the skill that answers the turn, and only a directive addressed
    // to the answer can claim it. Storing one on a directive scoped away from the
    // answer would look configured and do nothing, so it is rejected at authoring
    // rather than ignored at runtime.
    if (!addressesSurface(directive.surfaces, GENERATION_SURFACE.ANSWER)) {
      throw badRequest(
        `Directive binding requires the directive to apply to the agent's reply, but "${directive.name}" is scoped away from it`,
      );
    }
    const skill = await this.options.agentSkills?.findByName(workspaceId, agentId, binding.skillName);
    if (!skill) {
      throw badRequest(`Directive binding references unknown skill "${binding.skillName}"`);
    }
    if (!skill.enabled) {
      throw badRequest(`Directive binding skill "${binding.skillName}" is disabled`);
    }
    if (skill.invocationMode !== "agent_selectable") {
      throw badRequest(`Directive binding skill "${binding.skillName}" is not turn-selectable`);
    }
    // External MCP skills still claim the terminal turn; retrieve skills are
    // staged into the agentic answer loop as lookup tools. Action kinds settle
    // with outputs only, so they remain unsafe for directive binding.
    if (skill.kind !== "external_mcp" && skill.kind !== "retrieve") {
      throw badRequest(
        `Directive binding skill "${binding.skillName}" (kind "${skill.kind}") cannot answer chat turns or be staged as lookup; only external MCP and retrieve skills can be bound`,
      );
    }
  }

  private async checkCoherence(
    workspaceId: string,
    agent: AuthoredDirectiveAgentContext,
    candidate: NormalizedAuthoredDirectiveInput,
    existingDirectives: AuthoredDirective[],
  ): Promise<DirectiveCoherenceVerdict> {
    return this.runCoherenceCheck(
      candidate,
      buildCoherenceCheckInput(workspaceId, agent, authoredDirectiveToDirective(candidate), comparisonPool(existingDirectives)),
    );
  }

  /** Runs the checker on an input from `buildCoherenceCheckInput`, keeping its skip and fail-open rules. */
  private async runCoherenceCheck(
    candidate: NormalizedAuthoredDirectiveInput,
    input: DirectiveCoherenceCheckInput,
  ): Promise<DirectiveCoherenceVerdict> {
    // A disabled candidate is out of play and cannot fire, so it cannot conflict with
    // anything - skip the LLM round-trip entirely. This matters because disabling is the
    // emergency action an operator takes when a rule misfires; it must not cost a provider
    // call whose verdict the UI would discard. Re-enabling, in contrast, brings the
    // directive back into play and must still be checked, since that is exactly when a
    // reintroduced conflict would matter.
    if (!candidate.enabled) {
      return disabledCandidateVerdict();
    }
    try {
      return await this.options.coherenceChecker.check(input);
    } catch {
      return coherenceUnavailableVerdict();
    }
  }

  /**
   * The coherence inputs for a batch of candidates that will be created in order: each one is
   * compared with the agent's enabled directives, the built-in defaults, and the enabled
   * candidates before it, which is the set it would see if created after them. An earlier
   * candidate is referenced by `pendingDirectiveId(ref)` until it has a real id.
   */
  private async buildSequentialCoherenceInputs(
    workspaceId: string,
    agentId: string,
    candidates: ReadonlyArray<DirectiveCoherenceCandidate>,
  ): Promise<Array<{ candidate: NormalizedAuthoredDirectiveInput; input: DirectiveCoherenceCheckInput }>> {
    const agent = await this.requireAgent(workspaceId, agentId);
    const normalized = candidates.map((candidate) => ({ ref: candidate.ref, directive: this.validateInput(candidate.input) }));
    const existingDirectives = await this.options.repository.listDirectives(agentId, workspaceId);
    return normalized.map((candidate, index) => {
      const earlierCandidates = normalized
        .slice(0, index)
        .filter((earlier) => earlier.directive.enabled)
        .map((earlier) => ({ ...authoredDirectiveToDirective(earlier.directive), id: pendingDirectiveId(earlier.ref) }));
      return {
        candidate: candidate.directive,
        input: buildCoherenceCheckInput(
          workspaceId,
          agent,
          authoredDirectiveToDirective(candidate.directive),
          comparisonPool(existingDirectives, earlierCandidates),
        ),
      };
    });
  }

  /**
   * Checks candidates that will be created in order, without saving them, and returns their
   * verdicts with a token over every input the checker read. A caller that saves them later,
   * after re-reading the agent, compares `coherenceContextToken` with this token before passing
   * the verdicts to `create`. Each candidate's binding is validated as `create` would.
   */
  async assessCoherence(
    workspaceId: string,
    agentId: string,
    candidates: ReadonlyArray<DirectiveCoherenceCandidate>,
  ): Promise<DirectiveCoherenceAssessment> {
    const inputs = await this.buildSequentialCoherenceInputs(workspaceId, agentId, candidates);
    for (const { candidate } of inputs) {
      await this.validateBinding(workspaceId, agentId, candidate);
    }
    const verdicts: DirectiveCoherenceAssessment["verdicts"] = [];
    for (const [index, { candidate, input }] of inputs.entries()) {
      verdicts.push({ ref: candidates[index]?.ref ?? "", verdict: await this.runCoherenceCheck(candidate, input) });
    }
    return { verdicts, contextToken: coherenceContextTokenOf(inputs) };
  }

  /** The token `assessCoherence` would return for these candidates now, without calling the model. */
  async coherenceContextToken(
    workspaceId: string,
    agentId: string,
    candidates: ReadonlyArray<DirectiveCoherenceCandidate>,
  ): Promise<string> {
    return coherenceContextTokenOf(await this.buildSequentialCoherenceInputs(workspaceId, agentId, candidates));
  }

  private async requireAgent(workspaceId: string, agentId: string): Promise<AuthoredDirectiveAgentContext> {
    const agent = await this.options.repository.findByIdAndWorkspaceId(agentId, workspaceId);
    if (!agent) {
      throw notFound("Agent not found");
    }
    return agent;
  }
}
