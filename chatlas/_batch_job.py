from __future__ import annotations

import hashlib
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, Union

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ._chat import Chat
from ._content import Content
from ._provider import BatchStatus
from ._turn import AssistantTurn, Turn, user_turn
from ._typing_extensions import TypedDict

BatchStage = Literal["submitting", "waiting", "retrieving", "done"]


class BatchStateHash(TypedDict):
    provider: str
    model: str
    prompts: str
    user_turns: str


class BatchState(BaseModel):
    version: int
    stage: BatchStage
    batch: dict[str, Any]
    results: list[dict[str, Any]]
    started_at: int
    hash: BatchStateHash


ContentT = TypeVar("ContentT", bound=Union[str, Content])


class BatchJob:
    """
    Manages the lifecycle of a batch processing job.

    A batch job goes through several stages:
    1. "submitting" - Initial submission to the provider
    2. "waiting" - Waiting for processing to complete
    3. "retrieving" - Downloading results
    4. "done" - Processing complete
    """

    def __init__(
        self,
        chat: Chat,
        prompts: list[ContentT] | list[list[ContentT]],
        path: Union[str, Path],
        data_model: Optional[type[BaseModel]] = None,
        wait: bool = True,
    ):
        if not chat.provider.has_batch_support():
            raise ValueError("Batch requests are not supported by this provider")

        self.chat = chat
        self.prompts = prompts
        self.path = Path(path)
        self.data_model = data_model
        self.should_wait = wait

        # Convert prompts to user turns
        self.user_turns: list[Turn] = []
        for prompt in prompts:
            if not isinstance(prompt, (str, Content)):
                turn = user_turn(*prompt)
            else:
                turn = user_turn(prompt)
            self.user_turns.append(turn)

        # Job state management
        self.provider = chat.provider
        self.stage: BatchStage = "submitting"
        self.batch: dict[str, Any] = {}
        self.results: list[dict[str, Any]] = []

        # Load existing state if file exists and is not empty
        if self.path.exists() and self.path.stat().st_size > 0:
            self._load_state()
        else:
            self.started_at = time.time()

    def _load_state(self) -> None:
        with open(self.path, "r") as f:
            state = BatchState.model_validate_json(f.read())

        self.stage = state.stage
        self.batch = state.batch
        self.results = state.results
        self.started_at = state.started_at

        # Verify hash to ensure consistency
        stored_hash = state.hash
        current_hash = self._compute_hash()

        for key, value in current_hash.items():
            if stored_hash.get(key) != value:
                raise ValueError(
                    f"Batch state mismatch: {key} doesn't match stored value. "
                    f"Do you need to pick a different path?"
                )

    def _save_state(self) -> None:
        state = BatchState(
            version=1,
            stage=self.stage,
            batch=self.batch,
            results=self.results,
            started_at=int(self.started_at) if self.started_at else 0,
            hash=self._compute_hash(),
        )

        with open(self.path, "w") as f:
            f.write(state.model_dump_json(indent=2))

    def _compute_hash(self) -> BatchStateHash:
        turns = self.chat.get_turns(include_system_prompt=True)
        return {
            "provider": self.provider.name,
            "model": self.provider.model,
            "prompts": self._hash([str(p) for p in self.prompts]),
            "user_turns": self._hash([str(turn) for turn in turns]),
        }

    @staticmethod
    def _hash(x: Any) -> str:
        return hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()

    def step(self) -> bool:
        if self.stage == "submitting":
            return self._submit()
        elif self.stage == "waiting":
            return self._wait()
        elif self.stage == "retrieving":
            return self._retrieve()
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def step_until_done(self) -> Optional["BatchJob"]:
        while self.stage != "done":
            if not self.step():
                return None
        return self

    def _submit(self) -> bool:
        existing_turns = self.chat.get_turns(include_system_prompt=True)

        conversations = []
        for turn in self.user_turns:
            conversation = existing_turns + [turn]
            conversations.append(conversation)

        self.batch = self.provider.batch_submit(conversations, self.data_model)
        self.stage = "waiting"
        self._save_state()
        return True

    def _wait(self) -> bool:
        # Always poll once, even when wait=False
        status = self._poll()

        if self.should_wait:
            console = Console()

            with Progress(
                SpinnerColumn(),
                TextColumn("Processing..."),
                TextColumn("[{task.fields[elapsed]}]"),
                TextColumn("{task.fields[n_processing]} pending |"),
                TextColumn("[green]{task.fields[n_succeeded]}[/green] done |"),
                TextColumn("[red]{task.fields[n_failed]}[/red] failed"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "processing",
                    elapsed=self._elapsed(),
                    n_processing=status.n_processing,
                    n_succeeded=status.n_succeeded,
                    n_failed=status.n_failed,
                )

                while status.working:
                    time.sleep(0.5)
                    status = self._poll()
                    progress.update(
                        task,
                        elapsed=self._elapsed(),
                        n_processing=status.n_processing,
                        n_succeeded=status.n_succeeded,
                        n_failed=status.n_failed,
                    )

        if not status.working:
            self.stage = "retrieving"
            self._save_state()
            return True
        else:
            return False

    def _poll(self) -> "BatchStatus":
        if not self.batch:
            raise ValueError("No batch to poll")
        self.batch = self.provider.batch_poll(self.batch)
        self._save_state()
        return self.provider.batch_status(self.batch)

    def _elapsed(self) -> str:
        return str(timedelta(seconds=int(time.time()) - int(self.started_at)))

    def _retrieve(self) -> bool:
        if not self.batch:
            raise ValueError("No batch to retrieve")
        self.results = self.provider.batch_retrieve(self.batch)
        self.stage = "done"
        self._save_state()
        return True

    def result_turns(self) -> list[AssistantTurn | None]:
        turns = []
        for result in self.results:
            turn = self.provider.batch_result_turn(
                result, has_data_model=self.data_model is not None
            )
            if turn and turn.tokens is None and turn.completion:
                turn.tokens = self.provider.value_tokens(turn.completion)
            turns.append(turn)

        return turns
