#!/usr/bin/env python3
"""
spec/autospec.py — Auto-design a spec.md from a natural language prompt.

Without API key:  keyword-match → print the closest preset spec.md
With API key:     LLM reads preset specs as context → auto-designs a custom spec
                  then enters multi-round refinement loop

Supported keys (auto-detected in order):
    ANTHROPIC_API_KEY   → Claude
    OPENAI_API_KEY      → GPT-4o
    GEMINI_API_KEY      → Gemini

Usage:
    python3 spec/autospec.py "low-dose CT reconstruction with TV regularization"
    python3 spec/autospec.py "MRI mismatch correction with ESPIRiT"
    python3 spec/autospec.py "lensless imaging system design" --api-key sk-ant-...
    python3 spec/autospec.py "CT reconstruction" --save output.md
    OPENAI_API_KEY=sk-... python3 spec/autospec.py "photoacoustic speed-of-sound mismatch"
"""
import argparse, os, re, sys
from pathlib import Path

SPEC_DIR = Path(__file__).parent
sys.path.insert(0, str(SPEC_DIR))
from keyword_match import _match, MODALITIES   # reuse keyword matcher

# ── Spec context loader ───────────────────────────────────────────────────

def _load_preset_context(mod_id: str, prompt: str) -> dict[str, str]:
    """Load relevant preset specs as LLM context."""
    ctx = {}

    # Overview spec
    p = SPEC_DIR / f'{mod_id}.md'
    if p.exists():
        ctx['overview'] = p.read_text()

    # Infer use case from prompt
    pl = prompt.lower()
    if any(w in pl for w in ['mismatch', 'calibrat', 'correct', 'error', 'drift']):
        d = SPEC_DIR / '02_mismatch' / mod_id
        if d.exists():
            files = sorted(d.glob('*.md'))[:3]
            ctx['mismatch'] = '\n---\n'.join(f.read_text() for f in files)
    if any(w in pl for w in ['system', 'design', 'dag', 'pipeline', 'forward']):
        p = SPEC_DIR / '03_system_design' / f'{mod_id}.md'
        if p.exists():
            ctx['system_design'] = p.read_text()
    if any(w in pl for w in ['simulat', 'physics', 'equation', 'forward model']):
        for d in (SPEC_DIR / '04_simulation').iterdir():
            if mod_id[:3] in d.name or d.name[:3] in mod_id:
                p = d / 'spec.md'
                if p.exists():
                    ctx['simulation'] = p.read_text()
                    break

    # Default: reconstruct specs (top 3 CPU algorithms)
    if not ctx.get('mismatch') and not ctx.get('system_design'):
        d = SPEC_DIR / '01_reconstruct' / mod_id
        if d.exists():
            cpu_files = [f for f in sorted(d.glob('*.md'))
                         if '**CPU**' in f.read_text() or 'CPU' in f.read_text()][:3]
            if cpu_files:
                ctx['reconstruct'] = '\n---\n'.join(f.read_text() for f in cpu_files)

    return ctx


SYSTEM_PROMPT = """\
You are a physics-based imaging expert. Your job is to auto-design a PWM spec file.

A spec file is a minimal Markdown document (≤ 20 lines) with this exact format:
```
# {Modality} — {Algorithm or Task}

**CPU** or **GPU**  **PSNR**: ~XX dB  **Mismatch**: param `[range]`  *(optional reference)*
**Input**: measurement format
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/public/`

```python
from algorithm_base.{modality}.solvers import run_solver
# ... calibration or cfg if needed
x = run_solver('{solver_key}', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
```

Rules:
- Output ONLY the spec.md content. No prose, no explanation.
- Keep it ≤ 20 lines total.
- The Python block must be runnable copy-paste code.
- Use `run_solver()` from `algorithm_base.{modality}.solvers`.
- Base your design on the preset specs provided as context.
- If the user prompt asks for mismatch correction, include calibration code.
- If the user prompt asks for system design, include a short ASCII DAG above the code block.
"""


def _format_context(ctx: dict[str, str]) -> str:
    parts = []
    for label, content in ctx.items():
        parts.append(f"### Preset: {label}\n{content}")
    return '\n\n'.join(parts)


def _print_spec(spec: str, label: str = ''):
    print(f"\n{'─'*60}")
    if label:
        print(f"  {label}")
        print('─'*60)
    print(spec.strip())
    print('─'*60)


# ── No-API path ───────────────────────────────────────────────────────────

def run_preset(prompt: str):
    mod_id, disp_name = _match(prompt)
    if not mod_id:
        print("No match found. Try: python3 spec/autospec.py list")
        sys.exit(1)

    # Pick the most relevant preset spec
    pl = prompt.lower()
    if any(w in pl for w in ['mismatch', 'calibrat', 'correct', 'drift']):
        d = SPEC_DIR / '02_mismatch' / mod_id
        files = sorted(d.glob('*.md')) if d.exists() else []
        spec_file = files[0] if files else SPEC_DIR / f'{mod_id}.md'
        label = f'spec/02_mismatch/{mod_id}/{spec_file.name}'
    elif any(w in pl for w in ['system', 'design', 'pipeline', 'dag']):
        spec_file = SPEC_DIR / '03_system_design' / f'{mod_id}.md'
        label = f'spec/03_system_design/{mod_id}.md'
    elif any(w in pl for w in ['simulat', 'physics', 'equation']):
        sim_dirs = list((SPEC_DIR / '04_simulation').glob('*/spec.md'))
        match = next((p for p in sim_dirs if mod_id[:3] in p.parent.name), None)
        spec_file = match or SPEC_DIR / f'{mod_id}.md'
        label = str(spec_file.relative_to(SPEC_DIR.parent))
    else:
        spec_file = SPEC_DIR / f'{mod_id}.md'
        label = f'spec/{mod_id}.md'

    if not spec_file or not spec_file.exists():
        spec_file = SPEC_DIR / f'{mod_id}.md'
        label = f'spec/{mod_id}.md'

    print(f"\n[No API key — returning closest preset spec]")
    print(f"Query : {prompt!r}")
    print(f"Match : {disp_name}")
    _print_spec(spec_file.read_text(), label)
    print(f"\nTo auto-design a custom spec, set ANTHROPIC_API_KEY or pass --api-key")


# ── LLM backend (auto-detect provider) ───────────────────────────────────

def _detect_provider(api_key: str) -> tuple[str, str]:
    """Return (provider, key). Auto-detect from env if api_key not given."""
    if api_key:
        if api_key.startswith('sk-ant-'):
            return 'anthropic', api_key
        if api_key.startswith('sk-') or api_key.startswith('sk-proj-'):
            return 'openai', api_key
        return 'gemini', api_key
    # Auto-detect from environment
    for env, provider in [
        ('ANTHROPIC_API_KEY', 'anthropic'),
        ('OPENAI_API_KEY',    'openai'),
        ('GEMINI_API_KEY',    'gemini'),
    ]:
        v = os.environ.get(env, '')
        if v:
            return provider, v
    return '', ''


def _llm_call(provider: str, key: str, messages: list, system: str) -> str:
    """Single LLM call → returns reply text."""
    if provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model='claude-opus-4-6', max_tokens=1024,
            system=system, messages=messages,
        )
        return resp.content[0].text

    if provider == 'openai':
        import openai
        client = openai.OpenAI(api_key=key)
        full = [{'role': 'system', 'content': system}] + messages
        resp = client.chat.completions.create(
            model='gpt-4o-mini', max_tokens=1024, messages=full,
        )
        return resp.choices[0].message.content

    if provider == 'gemini':
        import urllib.request, json
        url = (f'https://generativelanguage.googleapis.com/v1beta/models/'
               f'gemini-2.5-flash:generateContent?key={key}')
        # Prepend system as first user turn for Gemini
        parts = [{'text': system + '\n\n---\n'}]
        for m in messages:
            parts.append({'text': f"[{m['role']}]: {m['content']}"})
        body = json.dumps({'contents': [{'parts': parts}]}).encode()
        req = urllib.request.Request(
            url, data=body, headers={'Content-Type': 'application/json'})
        resp = json.loads(urllib.request.urlopen(req).read())
        return resp['candidates'][0]['content']['parts'][0]['text']

    raise ValueError(f"Unknown provider: {provider}")


# ── API path ──────────────────────────────────────────────────────────────

def run_llm(prompt: str, api_key: str, save_path: Path | None):
    provider, key = _detect_provider(api_key)
    if not provider:
        print("No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")
        sys.exit(1)

    mod_id, disp_name = _match(prompt)
    if not mod_id:
        print("No modality match found. Try a more specific prompt.")
        sys.exit(1)

    print(f"\n[Provider: {provider} | Modality: {disp_name}]")
    ctx = _load_preset_context(mod_id, prompt)
    context_text = _format_context(ctx)

    messages = [
        {
            "role": "user",
            "content": (
                f"Design a spec.md for this request: {prompt!r}\n\n"
                f"Modality: {disp_name} (id: {mod_id})\n\n"
                f"Use these preset specs as reference:\n\n{context_text}"
            )
        }
    ]

    reply = _llm_call(provider, key, messages, SYSTEM_PROMPT)
    current = _extract_spec(reply)
    messages.append({"role": "assistant", "content": reply})

    _print_spec(current, f'Auto-designed: {mod_id}')

    if save_path:
        save_path.write_text(current, encoding='utf-8')
        print(f"Saved → {save_path}")

    # Refinement loop
    print("\nRefine the spec (or type 'save', 'show', 'quit'):\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'show':
            _print_spec(current)
            continue
        if user_input.lower() == 'save':
            out = save_path or Path(f'spec_{mod_id}_custom.md')
            out.write_text(current, encoding='utf-8')
            print(f"Saved → {out}")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = _llm_call(provider, key, messages, SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": reply})
        current = _extract_spec(reply)
        _print_spec(current)

        if save_path:
            save_path.write_text(current, encoding='utf-8')
            print(f"(auto-saved → {save_path})")


def _extract_spec(text: str) -> str:
    """Extract markdown spec from LLM reply (unwrap outer code fence if present)."""
    t = text.strip()
    # If wrapped in outer ``` fence (no language or 'markdown'), strip only that
    m = re.match(r'^```(?:markdown)?\n(.*)\n```$', t, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If starts with # heading directly, return as-is
    if t.startswith('#'):
        return t
    # Fallback: strip any single outer fence
    m2 = re.search(r'```(?:markdown)?\n(.*?)```', t, re.DOTALL)
    return m2.group(1).strip() if m2 else t


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Auto-design a PWM spec from a prompt.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 spec/autospec.py "CT reconstruction low-dose"
  python3 spec/autospec.py "MRI mismatch correction ESPIRiT"
  python3 spec/autospec.py "lensless system design" --api-key sk-ant-...
  python3 spec/autospec.py "photoacoustic speed mismatch" --save my_spec.md
  python3 spec/autospec.py list
        """
    )
    parser.add_argument('prompt', nargs='+', help='Natural language prompt or "list"')
    parser.add_argument('--api-key', default='',
                        help='API key (Anthropic/OpenAI/Gemini) — or set env var')
    parser.add_argument('--save', metavar='FILE',
                        help='Save resulting spec to this file')
    args = parser.parse_args()

    prompt = ' '.join(args.prompt)

    if prompt.lower() == 'list':
        print(f"{'ID':<30} Display Name")
        print('─' * 60)
        for mod_id, disp_name, _ in MODALITIES:
            print(f"{mod_id:<30} {disp_name}")
        return

    save_path = Path(args.save) if args.save else None

    if args.api_key:
        run_llm(prompt, args.api_key, save_path)
    else:
        run_preset(prompt)


if __name__ == '__main__':
    main()
