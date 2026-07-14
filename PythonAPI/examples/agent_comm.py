from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, wait, ALL_COMPLETED
from typing import Dict, List, Optional

# Reuse the OpenAI client and model config from llmutils to avoid duplicate initialization
from llmutil import llmutils as _llm


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Send message to another Agent for negotiation"
                "Used to inform the other party of your draft plan, occupied nodes, or negotiation request."
                "Continue reasoning after sending, and wait for the other party’s reply in the next round."
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
    """
    All agents share the same Mailbox instance.
    It maintains an independent message queue for each agent; all operations
    are locked to guarantee thread safety.
    """

    def __init__(self, message_interceptor=None):
        self._lock = threading.Lock()
        self._queues: Dict[str, List[Dict]] = defaultdict(list)
        self._interceptor = message_interceptor

    def send(self, from_agent: str, to_agent: str, content: str) -> str:
        """
        Put a message into the target agent's mailbox.
        If a message_interceptor is registered, call it to obtain extra feedback
        text (e.g. collision check results) and append it to the end of the
        message before delivering to the receiver.
        Interceptor signature: (from_agent: str, to_agent: str, content: str) -> str
        Returns the extra text produced by the interceptor (e.g. a collision
        report) so the caller can write it back into the tool result.
        """
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
        print(f"[Router] {from_agent} -> {to_agent}: {content[:1000]}")
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

    MAX_TURNS = 50

    def __init__(self, name: str, system_prompt: str, mailbox: Mailbox,
                 finalization_guard=None):
        self.name     = name
        self.mailbox  = mailbox
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix=f"AgentRunner-{name}")
        self._messages: List[Dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._finalization_guard = finalization_guard
        self.iterations: int = 0
        self.total_llm_time: float = 0.0


    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def run_async(self) -> Future:
        return self._executor.submit(self._run_loop)

    def _run_loop(self) -> str:
        for turn in range(self.MAX_TURNS):
            self._inject_inbox()
            print(f"[{self.name}]  {turn + 1} th iteration...")
            _t0 = time.time()
            response = _llm.openai_client.chat.completions.create(
                model=_llm.ENGINE,
                messages=self._messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0,
                seed=0,
            )
            self.total_llm_time += time.time() - _t0
            self.iterations = turn + 1

            raw_msg = response.choices[0].message
            msg_dict: Dict = {"role": "assistant", "content": raw_msg.content or ""}
            if raw_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    tc.model_dump() for tc in raw_msg.tool_calls
                ]
            self._messages.append(msg_dict)
            if not raw_msg.tool_calls:
                if self._finalization_guard:
                    block_msg = self._finalization_guard(self.name, raw_msg.content or "")
                    if block_msg:
                        self._messages.append({"role": "user", "content": block_msg})
                        print(f"[{self.name}] Finalization blocked by guard, continuing negotiation...")
                        continue
                print(f"[{self.name}] Reasoning completed")
                return raw_msg.content or ""

            for tc in raw_msg.tool_calls:
                tool_result = self._handle_tool_call(tc)
                self._messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result,
                })
                print("result: " + tool_result)

        print(f"[{self.name}] reached max iteration {self.MAX_TURNS}, forcing return")
        for m in reversed(self._messages):
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                return m["content"]
        return ""

    def _inject_inbox(self) -> None:
        for msg in self.mailbox.receive_all(self.name):
            text = f"[from {msg['from']}]: {msg['content']}"
            print(f"[{self.name}] got message <- {msg['from']} as {msg['content']}")
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
    """
    Manage a group of AgentRunners sharing the same Mailbox.

    Typical usage (integrated with the CARLA main loop)
    ---------------------------------------------------
        pool = AgentPool()
        pool.add_agent("agent1", system_prompt_1)
        pool.add_agent("agent2", system_prompt_2)
        pool.send_task("agent1", user_message_1)
        pool.send_task("agent2", user_message_2)

        # Non-blocking start, returns a Future
        future = pool.run_all_async()

        # Poll in the CARLA main loop
        if future.done():
            results = future.result()
            path1 = results["agent1"]   # final LLM reply text
            path2 = results["agent2"]
    """

    def __init__(self, message_interceptor=None, finalization_guard=None):
        self.mailbox  = Mailbox(message_interceptor=message_interceptor)
        self._agents: Dict[str, AgentRunner] = {}
        self._finalization_guard = finalization_guard
        self._outer_pool: Optional[ThreadPoolExecutor] = None

    def add_agent(self, name: str, system_prompt: str) -> None:
        self._agents[name] = AgentRunner(name, system_prompt, self.mailbox,
                                         finalization_guard=self._finalization_guard)

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

    def get_stats(self) -> dict:
        return {
            name: {
                "iterations":     agent.iterations,
                "total_llm_time": agent.total_llm_time,
            }
            for name, agent in self._agents.items()
        }
