"""Shared Pydantic v2 models imported by every layer of the workflow engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ToolSpec",
    "PlanStep",
    "ExecutionPlan",
    "ToolResult",
    "RetrievedPlan",
    "DraftResponse",
    "GuardVerdict",
    "AgentState",
]


class ToolSpec(BaseModel):
    """Capability descriptor for a single MCP tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool name as exposed by the MCP server.")
    description: str = Field(description="Human-readable summary of what the tool does.")
    server: str = Field(description="MCP server key (calendar | gmail | notion | slack).")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for the tool's input arguments."
    )


class PlanStep(BaseModel):
    """One executable step inside an ExecutionPlan."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable step identifier used for DAG references.")
    tool: str = Field(description="MCP server key that owns the tool.")
    action: str = Field(description="Specific tool action to invoke on the server.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed verbatim to the tool."
    )
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of steps that must complete before this one."
    )


class ExecutionPlan(BaseModel):
    """Typed plan produced by the planner node."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="Short plain-language rationale for the plan.")
    steps: list[PlanStep] = Field(description="Ordered or DAG-shaped list of plan steps.")
    strategy: Literal["sequential", "parallel", "mixed"] = Field(
        description="Execution strategy for the orchestrator."
    )
    complexity_score: int = Field(ge=1, le=10, description="Planner self-rated complexity 1-10.")
    estimated_cost_usd: float = Field(ge=0.0, description="Rough estimated total cost in USD.")
    requires_approval: bool = Field(description="True if planner thinks HITL approval is needed.")


class ToolResult(BaseModel):
    """Outcome of a single tool invocation against an MCP server."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(description="ID of the PlanStep this result belongs to.")
    ok: bool = Field(description="True if the tool call succeeded.")
    output: dict[str, Any] | None = Field(default=None, description="Tool output payload.")
    error: str | None = Field(default=None, description="Error message if ok is False.")
    latency_ms: int = Field(ge=0, description="Round-trip latency in milliseconds.")


class RetrievedPlan(BaseModel):
    """Past plan returned by the agentic RAG retriever."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="Qdrant point ID for the past plan.")
    request_text: str = Field(description="Original user request that produced the plan.")
    plan_json: dict[str, Any] = Field(description="Serialized ExecutionPlan for the past run.")
    summary: str = Field(description="Short human-readable summary of the past plan.")
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity to current request.")
    fused_score: float = Field(
        default=0.0,
        description="RRF-fused rank score used only for ordering; not in cosine space, "
        "not comparable across queries, and never fed into the guardrails formula.",
    )


class DraftResponse(BaseModel):
    """User-facing draft assembled by the response composer."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="One-line summary of what was done or proposed.")
    detail_markdown: str = Field(description="Full markdown body shown to the user.")
    actions_taken: list[str] = Field(description="Concrete successful actions.")
    actions_pending: list[str] = Field(description="Actions that failed or need follow-up.")
    citations: list[str] = Field(default_factory=list, description="Sources backing the draft.")


class GuardVerdict(BaseModel):
    """Structured judgement produced by the guard-judge LLM."""

    model_config = ConfigDict(extra="forbid")

    tone_fit: float = Field(ge=0.0, le=1.0, description="Alignment with the user's tone.")
    hallucination_risk: float = Field(ge=0.0, le=1.0, description="Probability of fabricated content.")
    instruction_adherence: float = Field(ge=0.0, le=1.0, description="How well it followed the request.")


class AgentState(BaseModel):
    """Central LangGraph state shared by every orchestration node."""

    model_config = ConfigDict(extra="ignore")

    trace_id: str = Field(description="Unique trace ID for this workflow run.")
    user_id: str = Field(description="Authenticated user ID.")
    session_id: str = Field(description="Chat session ID.")
    user_request: str = Field(description="Original natural-language request.")
    retrieved_plans: list[RetrievedPlan] = Field(default_factory=list, description="RAG hits.")
    plan: ExecutionPlan | None = Field(default=None, description="Current execution plan.")
    tool_results: list[ToolResult] = Field(default_factory=list, description="Tool call outcomes.")
    draft: DraftResponse | None = Field(default=None, description="Composed draft response.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Guard-computed confidence.")
    requires_approval: bool = Field(default=False, description="Whether HITL is required.")
    approval_token: str | None = Field(default=None, description="Token issued for HITL resume.")
    retry_count: int = Field(default=0, ge=0, description="Number of planner retries used.")
    error: str | None = Field(default=None, description="Terminal error code if any.")
