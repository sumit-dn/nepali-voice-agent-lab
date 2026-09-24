"""FAKE, local-only tools for tool-calling experiments. They touch no real system and hold synthetic data."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from voice_lab.providers.base import LLMProvider, LLMResult

_CUSTOMERS = {"9841234567": {"customer_id": "C-1001", "name": "राम बहादुर थापा", "phone": "9841234567"}}
_BALANCES = {"C-1001": 25000}
_APPOINTMENTS = {"A-501": {"customer_id": "C-1001", "date": "2026-10-01", "time": "10:00"}}


def get_customer(phone: str) -> dict[str, Any]:
    digits = "".join(c for c in str(phone) if c.isdigit())[-10:]
    return _CUSTOMERS.get(digits, {"error": "customer_not_found"})


def get_balance(customer_id: str) -> dict[str, Any]:
    if customer_id not in _BALANCES:
        return {"error": "customer_not_found"}
    return {"customer_id": customer_id, "balance": _BALANCES[customer_id], "currency": "NPR"}


def check_availability(date: str) -> dict[str, Any]:
    return {"date": date, "available_times": ["10:00", "14:00", "16:30"]}


def book_appointment(customer_id: str, date: str, time: str) -> dict[str, Any]:
    return {"status": "booked", "appointment_id": "A-777", "customer_id": customer_id, "date": date, "time": time}


def cancel_appointment(appointment_id: str) -> dict[str, Any]:
    return (
        {"status": "cancelled", "appointment_id": appointment_id}
        if appointment_id in _APPOINTMENTS
        else {"error": "not_found"}
    )


def transfer_to_human(reason: str) -> dict[str, Any]:
    return {"status": "transfer_queued", "reason": reason}


def _schema(fn: Callable[..., Any], description: str, **props: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
                "required": list(props),
            },
        },
    }


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    f.__name__: f
    for f in (get_customer, get_balance, check_availability, book_appointment, cancel_appointment, transfer_to_human)
}
SCHEMAS = [
    _schema(get_customer, "Look up a customer by phone number.", phone="10-digit phone number, ASCII digits"),
    _schema(get_balance, "Get account balance for a customer.", customer_id="customer id, e.g. C-1001"),
    _schema(check_availability, "List free appointment times on a date.", date="date as YYYY-MM-DD (AD)"),
    _schema(
        book_appointment, "Book an appointment.", customer_id="customer id", date="YYYY-MM-DD (AD)", time="HH:MM 24h"
    ),
    _schema(cancel_appointment, "Cancel an appointment.", appointment_id="appointment id, e.g. A-501"),
    _schema(transfer_to_human, "Transfer the caller to a human agent.", reason="short reason"),
]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown_tool:{name}"}
    try:
        return fn(**arguments)
    except TypeError as e:  # wrong/missing arguments are a model error worth recording
        return {"error": f"bad_arguments: {e}"}


def run_with_tools(
    llm: LLMProvider, messages: list[dict[str, Any]], max_steps: int = 4
) -> tuple[LLMResult, list[dict[str, Any]], float | None, float]:
    """Model -> tool -> model loop. Returns (final result, tool trace, first-call TTFT, total seconds)."""
    msgs = list(messages)
    trace: list[dict[str, Any]] = []
    ttft, total = None, 0.0
    for _ in range(max_steps):
        result = llm.generate(msgs, tools=SCHEMAS)
        ttft = result.time_to_first_token if ttft is None else ttft
        total += result.latency_seconds
        if not result.tool_calls:
            break
        msgs.append(
            {
                "role": "assistant",
                "content": result.text or "",
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                    }
                    for c in result.tool_calls
                ],
            }
        )
        for c in result.tool_calls:
            output = call_tool(c["name"], c["arguments"])
            trace.append({**c, "result": output})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(output, ensure_ascii=False)})
    return result, trace, ttft, total
