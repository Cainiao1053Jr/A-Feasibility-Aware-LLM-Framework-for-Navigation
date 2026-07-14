from __future__ import annotations

import json
import os
import datetime
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, wait, ALL_COMPLETED
from typing import Dict, List, Optional

from llmutil import llmutils as _llm


class _Logger:

    def __init__(self, log_path: str):
        self._path = log_path
        self._lock = threading.Lock()
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=== Multi-Agent Session Log ===\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def write(self, tag: str, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}][{tag}] {msg}\n"
        with self._lock:
            with open(self._path, 'a', encoding='utf-8') as f:
                f.write(entry)

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Send message to another Agent for negotiation"
                "Used to inform the other party of your draft plan, occupied nodes, or negotiation request."
                "Continue reasoning after sending, and wait for the other party's reply in the next round."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Target Agent name such as 'agent1' or 'agent2'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Message body describing the planning intent, draft path, or negotiation details.",
                    },
                },
                "required": ["to", "content"],
            },
        },
    },
]


class Mailbox:

    def __init__(self, message_interceptor=None, logger: "_Logger | None" = None):
        self._lock = threading.Lock()
        self._queues: Dict[str, List[Dict]] = defaultdict(list)
        self._interceptor = message_interceptor
        self._logger = logger

    def send(self, from_agent: str, to_agent: str, content: str) -> str:
        extra = ""
        if self._interceptor is not None:
            try:
                extra = self._interceptor(from_agent, to_agent, content) or ""
            except Exception as exc:
                print(f"[Router] interceptor error: {exc}")
        full_content = content if not extra else f"{content}\n\n{extra}"
        with self._lock:
            self._queues[to_agent].append({
                "from":      from_agent,
                "content":   full_content,
                "timestamp": time.time(),
            })
        print(f"[Router] {from_agent} → {to_agent}: {content[:1000]}")
        if self._logger:
            self._logger.write("Router", f"{from_agent} → {to_agent}:\n{full_content}")
        return extra

    def receive_all(self, agent_name: str) -> List[Dict]:
        with self._lock:
            msgs = list(self._queues[agent_name])
            self._queues[agent_name].clear()
        return msgs

    def has_messages(self, agent_name: str) -> bool:
        with self._lock:
            return bool(self._queues[agent_name])


class AgentRunner:
    MAX_TURNS = 20

    def __init__(self, name: str, system_prompt: str, mailbox: Mailbox,
                 finalization_guard=None, logger: "_Logger | None" = None):
        self.name     = name
        self.mailbox  = mailbox
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix=f"AgentRunner-{name}")
        self._messages: List[Dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._finalization_guard = finalization_guard
        self._logger = logger


    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def run_async(self) -> Future:
        return self._executor.submit(self._run_loop)


    def _run_loop(self) -> str:
        for turn in range(self.MAX_TURNS):
            self._inject_inbox()

            print(f"[{self.name}]  {turn + 1} th iteration...")
            if self._logger:
                self._logger.write(self.name, f"── Turn {turn + 1} ──────────────────────────────────────")
            response = _llm.openai_client.chat.completions.create(
                model=_llm.ENGINE,
                messages=self._messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0,
                seed=0,
            )

            raw_msg = response.choices[0].message
            msg_dict: Dict = {"role": "assistant", "content": raw_msg.content or ""}
            if raw_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    tc.model_dump() for tc in raw_msg.tool_calls
                ]
            self._messages.append(msg_dict)
            if self._logger:
                self._logger.write(self.name, f"LLM Response:\n{raw_msg.content or ''}")
            if not raw_msg.tool_calls:
                if self._finalization_guard:
                    block_msg = self._finalization_guard(self.name, raw_msg.content or "")
                    if block_msg:
                        self._messages.append({"role": "user", "content": block_msg})
                        print(f"[{self.name}] Finalization blocked by guard, continuing negotiation...")
                        if self._logger:
                            self._logger.write(self.name, f"Finalization blocked:\n{block_msg}")
                        continue
                print(f"[{self.name}] Reasoning completed")
                if self._logger:
                    self._logger.write(self.name, f"Final Answer:\n{raw_msg.content or ''}")
                return raw_msg.content or ""

            for tc in raw_msg.tool_calls:
                if self._logger:
                    self._logger.write(self.name, f"Tool Call: {tc.function.name}  args={tc.function.arguments}")
                tool_result = self._handle_tool_call(tc)
                self._messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result,
                })
                print("result: " + tool_result)
                if self._logger:
                    self._logger.write(self.name, f"Tool Result: {tool_result}")

        print(f"[{self.name}] reached max iteration {self.MAX_TURNS}, forcing return")
        if self._logger:
            self._logger.write(self.name, f"Max turns ({self.MAX_TURNS}) reached, forcing return")
        for m in reversed(self._messages):
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                if self._logger:
                    self._logger.write(self.name, f"Forced Final Answer:\n{m['content']}")
                return m["content"]
        return ""

    def _inject_inbox(self) -> None:
        for msg in self.mailbox.receive_all(self.name):
            text = f"[from {msg['from']}]: {msg['content']}"
            print(f"[{self.name}] got message ← {msg['from']} as {msg['content']}")
            if self._logger:
                self._logger.write(self.name, f"Inbox ← {msg['from']}:\n{msg['content']}")
            self._messages.append({"role": "user", "content": text})

    def _handle_tool_call(self, tc) -> str:
        fn = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            return '{"error": "failed to decode"}'

        if fn == "send_message":
            to      = args.get("to", "")
            content = args.get("content", "")
            if not to or not content:
                return '{"error": "missing to or content field"}'
            extra = self.mailbox.send(self.name, to, content)
            result: Dict = {"status": "ok", "to": to}
            if extra:
                result["collision_report"] = extra
            return json.dumps(result, ensure_ascii=False)

        return json.dumps({"error": f"unknown tool: {fn}"}, ensure_ascii=False)


class AgentPool:
    def __init__(self, message_interceptor=None, finalization_guard=None,
                 log_dir: str = None):
        logger = _Logger(os.path.join(log_dir, "agent_log.txt")) if log_dir else None
        self.mailbox  = Mailbox(message_interceptor=message_interceptor, logger=logger)
        self._agents: Dict[str, AgentRunner] = {}
        self._finalization_guard = finalization_guard
        self._logger = logger
        self._outer_pool: Optional[ThreadPoolExecutor] = None

    def add_agent(self, name: str, system_prompt: str) -> None:
        self._agents[name] = AgentRunner(name, system_prompt, self.mailbox,
                                         finalization_guard=self._finalization_guard,
                                         logger=self._logger)

    def send_task(self, agent_name: str, user_message: str) -> None:
        if agent_name not in self._agents:
            raise KeyError(f"Agent '{agent_name}' Registration invalid, please call add_agent() first.")
        self._agents[agent_name].add_user_message(user_message)

    def run_all(self, timeout: float = 180.0) -> Dict[str, str]:
        futures: Dict[str, Future] = {
            name: agent.run_async()
            for name, agent in self._agents.items()
        }

        done, not_done = wait(futures.values(), timeout=timeout,
                              return_when=ALL_COMPLETED)

        results: Dict[str, str] = {}
        for name, fut in futures.items():
            if fut in done:
                try:
                    results[name] = fut.result()
                except Exception as e:
                    print(f"[AgentPool] {name} Reasoning failed: {e}")
                    results[name] = ""
            else:
                print(f"[AgentPool] {name} Time expires, returning empty result")
                fut.cancel()
                results[name] = ""

        return results

    def run_all_async(self, timeout: float = 180.0) -> Future:
        self._outer_pool = ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix="AgentPool-outer")
        return self._outer_pool.submit(self.run_all, timeout)
