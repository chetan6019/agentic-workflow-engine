from langchain_core.messages import SystemMessage, HumanMessage
from app.prompts import (
    PROMPT_VERSION, fingerprint,
    build_planner_messages
)
from app.core.state import ExecutionPlan, GuardVerdict, DraftResponse

def test_planner_messages_render_and_fingerprint():
    msgs = build_planner_messages(
        user_request="hi", examples=[], tool_specs=[],
        now_local="2026-06-24T10:00:00+05:30", tz_name="Asia/Kolkata",
        weekday_ref="Monday: 2026-06-30",
    )
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)
    assert PROMPT_VERSION in PROMPT_VERSION   # tautology — pin existence
    assert len(fingerprint(msgs)) == 12

# Pin the planner's documented JSON example against ExecutionPlan.
def test_planner_example_validates():
    canonical = {
        "reasoning": "User asked to send an email.",
        "steps": [{"id":"s1","tool":"google","action":"send_email",
                   "arguments":{"to":"a@b.com","subject":"Hi","body":"Hello"},
                   "depends_on":[]}],
        "strategy": "sequential",
        "complexity_score": 1, "estimated_cost_usd": 0.0,
        "requires_approval": False,
    }
    ExecutionPlan.model_validate(canonical)        # must not raise

def test_guard_example_validates():
    GuardVerdict.model_validate(
        {"tone_fit": 0.9, "hallucination_risk": 0.1, "instruction_adherence": 0.95}
    )

def test_composer_example_validates():
    DraftResponse.model_validate({
        "summary": "Sent the email.",
        "detail_markdown": "I sent your email to a@b.com.",
        "actions_taken": ["Sent email"],
        "actions_pending": [], "citations": [],
    })