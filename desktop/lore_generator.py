"""
lore_generator.py

Takes a batch of raw tagged records (object tags + GPS, produced in the
field) and calls an LLM to turn them into structured NPCs, location
fixtures, and lore fragments, matching the schema the existing MMO
world-database pipeline already expects.

Supports two backends, configurable in config.yaml:
    - local_llama_server: a self-hosted llama.cpp server (llama-server)
      running on any machine on your LAN, talked to over its
      OpenAI-compatible /v1/chat/completions endpoint
    - ollama: local inference via Ollama (fallback / alternative)
"""

import json
from typing import List, Dict, Optional
from schema import NPCRecord, LocationFixture, LoreFragment


def extract_json_object(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from LLM output.

    Small local models routinely wrap the JSON in ```json fences or add a
    sentence of prose around it despite the "ONLY the JSON" instruction.
    Try the raw string first, then the first balanced {...} span. Returns
    None if nothing parses to a dict.
    """
    for candidate in (raw, _first_braced_span(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_braced_span(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None

PROMPT_TEMPLATE = """You are helping generate fantasy MMO world content from
real-world location tags collected on a walk. You will be given a list of
object tags (already translated to fantasy equivalents) observed at a single
location cluster, plus optional movement/gait impressions of people
encountered there. Generate a JSON object with this exact structure:

{{
  "zone_name": "<evocative fantasy name for this location>",
  "npcs": [{{"name": "...", "role": "...", "description": "..."}}],
  "fixtures": [{{"name": "...", "fixture_type": "...", "description": "..."}}],
  "lore": [{{"title": "...", "text": "..."}}]
}}

Keep NPC count to 0-2 and fixtures to 1-3 per cluster. This feeds a living
world system where density should build up gradually over many walks, not
all at once. If gait impressions are provided, use them as loose character
flavor for NPCs (posture, gait, bearing) rather than literal descriptions.
These are impressions of movement style only, not descriptions of any real
individual.

Fantasy tags observed at this location:
{tags}

Gait/movement impressions observed at this location (flavor only):
{gait}

Respond with ONLY the JSON object, no other text.
"""


class LoreGenerator:
    def __init__(self, provider: str, local_llama_server_cfg: dict = None, ollama_cfg: dict = None):
        self.provider = provider
        self.local_llama_server_cfg = local_llama_server_cfg or {}
        self.ollama_cfg = ollama_cfg or {}

    def _call_local_llama_server(self, prompt: str) -> str:
        """
        Talks to llama-server's OpenAI-compatible /v1/chat/completions
        endpoint on whatever machine you're running it on. No API key
        is needed since it's a LAN call. Start the server with something
        like:
            llama-server -m qwen2.5-14b-instruct.gguf --host 0.0.0.0 --port 8080 -ngl 99
        (adjust -ngl / model path to whatever you're already running.)
        """
        import requests
        host = self.local_llama_server_cfg.get("host", "http://localhost:8080")
        timeout = self.local_llama_server_cfg.get("timeout_s", 120)
        resp = requests.post(
            f"{host}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 1000,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, prompt: str) -> str:
        import requests
        timeout = self.ollama_cfg.get("timeout_s", 120)
        resp = requests.post(
            f"{self.ollama_cfg.get('host', 'http://localhost:11434')}/api/generate",
            json={"model": self.ollama_cfg.get("model", "mistral:7b"),
                  "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def generate_for_cluster(self, fantasy_tags: List[str], zone_id: str,
                              center_lat: float, center_lon: float,
                              source_timestamps: List[float],
                              gait_descriptors: List[str] = None) -> tuple:
        """Returns (zone_name, npcs, fixtures, lore) for one location cluster."""
        gait_descriptors = gait_descriptors or []
        prompt = PROMPT_TEMPLATE.format(
            tags=", ".join(sorted(set(fantasy_tags))),
            gait=", ".join(sorted(set(gait_descriptors))) if gait_descriptors else "none observed",
        )

        if self.provider == "local_llama_server":
            raw = self._call_local_llama_server(prompt)
        elif self.provider == "ollama":
            raw = self._call_ollama(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        parsed = extract_json_object(raw)
        if parsed is None:
            # Model didn't return usable JSON even after fence/prose
            # stripping. Fail soft with an empty result rather than
            # crashing the whole batch.
            print(f"[LoreGenerator] Failed to parse LLM output for zone {zone_id}, skipping.")
            return (f"Unnamed Zone {zone_id}", [], [], [])

        # Valid JSON can still be missing fields or have the wrong shape
        # (routine from small local models). Fail soft per item: drop the
        # malformed entry, keep the rest of the zone and the batch.
        npcs, fixtures, lore = [], [], []
        for n in parsed.get("npcs", []) or []:
            try:
                npcs.append(NPCRecord(name=n["name"], role=n["role"],
                                      description=n["description"],
                                      source_fantasy_tags=fantasy_tags,
                                      source_gait_traits=gait_descriptors,
                                      zone_id=zone_id))
            except (KeyError, TypeError) as e:
                print(f"[LoreGenerator] Skipping malformed NPC entry in zone {zone_id}: {e}")

        for f in parsed.get("fixtures", []) or []:
            try:
                fixtures.append(LocationFixture(name=f["name"],
                                                fixture_type=f["fixture_type"],
                                                description=f["description"],
                                                gps_lat=center_lat, gps_lon=center_lon,
                                                zone_id=zone_id))
            except (KeyError, TypeError) as e:
                print(f"[LoreGenerator] Skipping malformed fixture entry in zone {zone_id}: {e}")

        for l in parsed.get("lore", []) or []:
            try:
                lore.append(LoreFragment(title=l["title"], text=l["text"],
                                         related_zone_id=zone_id,
                                         source_record_timestamps=source_timestamps))
            except (KeyError, TypeError) as e:
                print(f"[LoreGenerator] Skipping malformed lore entry in zone {zone_id}: {e}")

        zone_name = parsed.get("zone_name") or f"Unnamed Zone {zone_id}"
        return (zone_name, npcs, fixtures, lore)
