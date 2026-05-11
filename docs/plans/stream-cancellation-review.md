# Stream Cancellation Branch — Review Findings

## 1. `_as_controller` called redundantly at every layer

`_as_controller(controller)` is called in `stream()`, `_chat_impl()`, `_submit_turns()` — up to four times per request. The function has a side effect (`_ensure_ready` warns + resets), so calling it multiple times is harmless but confusing.

**Suggestion:** Call `_as_controller` once at the public entry points (`stream`, `stream_async`, `chat`, `chat_async`). Internal methods accept a non-optional `StreamController` — they're never called externally and always receive one.

**Status:** [x] — `_as_controller` called once at `stream()`/`stream_async()`/`chat()`/`chat_async()`. Internal methods now take `controller: StreamController` (keyword-only, non-optional).

## 2. `TurnAccumulator` directly mutates `Chat._turns`

The accumulator receives `self._turns` (a mutable list) and modifies it in three different methods (`begin_turn`, `complete_turn`, `finalize_turn`). The index-based replacement (`self._turns[self._turn_idx] = turn`) assumes nothing else touches the list between `begin_turn` and `complete_turn`.

**Suggestion:** Have `TurnAccumulator` own the partial turn internally and return the final `(user_turn, assistant_turn)` pair for the caller to append. Eliminates index tracking and list-mutation-at-a-distance.

```python
acc = TurnAccumulator(controller)
acc.begin(user_turn)
# ... streaming loop ...
result = acc.finalize()  # returns (user_turn, AssistantTurn)
self._turns.extend(result)
```

**Status:** [ ]

## 3. Partial turn filtering in `token_usage_to_turns` is fragile

The paired-skip `while` loop assumes partial turns are always preceded by a user turn and always in pairs. The same "exclude partials" logic also appears separately in `token_cost()` and `__repr__()`.

**Suggestion:** A single helper or just skip `is_partial` assistant turns inline — no need to also skip the preceding user turn if you're only aggregating assistant turn data.

**Status:** [x] — Extracted `_complete_assistant_turns()` and `_complete_turn_pairs()` helpers on `Chat`. All three call sites now use them.

## 4. Non-streaming path doesn't use `TurnAccumulator`

The `else` branch in `_submit_turns` still directly appends to `self._turns`. Streaming and non-streaming now use different mechanisms for turn management.

**Status:** [ ] (may be intentional — decide explicitly)

## 5. Response closing logic in `finally` is duck-typing through `Any`

The async `_submit_turns_async` finally block has a `hasattr` cascade through `Any` to close the response. This should be captured in the `Provider` protocol or at minimum extracted into a helper.

**Suggestion:** Extract into `_close_response(response)` / `_aclose_response(response)` so the `finally` block stays clean.

**Status:** [x] — Extracted `_close_response()` and `_aclose_response()` helpers. Both `finally` blocks now call single-line helpers.

## 6. `StreamController._ensure_ready` — pick a lane

Auto-reset with warning is a middle ground that serves neither goal. If it's a convenience, auto-reset silently. If it's a bug, raise an error.

**Status:** [ ]

## 7. `resolve_assistant_turn` mutates its argument

Takes a `Turn`, mutates it, returns it as `AssistantTurn`. The name suggests "resolve" (compute/derive), not "mutate." The fact that it returns the same object makes it look like a pure transform.

**Suggestion:** Rename to `finalize_assistant_turn` or similar to signal mutation.

**Status:** [x] — Renamed to `finalize_assistant_turn`.
