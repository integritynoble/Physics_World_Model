#!/usr/bin/env python3
"""
spec/refine.py — LLM-assisted spec refinement.

Without API key: prints the preset spec.
With API key:    multi-round prompting to customize the spec.

Usage:
    python3 spec/refine.py spec/01_reconstruct/ct/pnp_admm_nlm.md
    python3 spec/refine.py spec/ct.md --api-key sk-ant-...
    python3 spec/refine.py spec/03_system_design/mri.md --save
    ANTHROPIC_API_KEY=sk-ant-... python3 spec/refine.py spec/ct.md
"""
import argparse, os, sys
from pathlib import Path


SYSTEM_PROMPT = """\
You are a physics-based imaging expert helping a user customize a PWM spec file.

A spec file is a minimal Markdown document with:
- A title line (# Modality — Algorithm)
- One-line metadata (CPU/GPU, PSNR, mismatch param)
- A Python run-button code block using `run_solver()`

Rules:
- Keep specs short (≤ 20 lines). Do NOT add explanations or prose.
- Only output the complete updated spec.md content — no commentary before or after.
- The run button must stay runnable copy-paste Python.
- Preserve the original header format.
"""


def load_spec(path: Path) -> str:
    if not path.exists():
        sys.exit(f"Error: {path} not found.\nTry: python3 spec/keyword_match.py list")
    return path.read_text(encoding='utf-8')


def print_spec(spec: str, path: Path):
    print(f"\n{'─'*60}")
    print(f"  {path}")
    print('─'*60)
    print(spec)
    print('─'*60)


def refine_loop(spec: str, path: Path, api_key: str, save: bool):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    history = [{"role": "user", "content": f"Here is the current spec:\n\n```markdown\n{spec}\n```"}]

    # Prime with the initial spec as assistant context
    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=history + [{"role": "user", "content": "Acknowledge you have the spec. Reply only: 'Ready. What would you like to change?'"}],
    )
    print("\nLLM ready. Type your request, or 'show' / 'save' / 'quit'.\n")

    current = spec
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'show':
            print_spec(current, path)
            continue
        if user_input.lower() == 'save':
            path.write_text(current, encoding='utf-8')
            print(f"Saved → {path}")
            continue

        history.append({"role": "user", "content": user_input})
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Current spec:\n\n```markdown\n{current}\n```"},
                *history,
            ],
        )
        reply = resp.content[0].text.strip()
        history.append({"role": "assistant", "content": reply})

        # Extract markdown block if wrapped
        if "```" in reply:
            import re
            m = re.search(r'```(?:markdown)?\n(.*?)```', reply, re.DOTALL)
            if m:
                current = m.group(1).strip()
            else:
                current = reply
        else:
            current = reply

        print_spec(current, path)
        if save:
            path.write_text(current, encoding='utf-8')
            print(f"(auto-saved)")


def main():
    parser = argparse.ArgumentParser(description='View or refine a PWM spec with LLM.')
    parser.add_argument('spec', help='Path to spec .md file, e.g. spec/ct.md')
    parser.add_argument('--api-key', default=os.environ.get('ANTHROPIC_API_KEY', ''),
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--save', action='store_true',
                        help='Auto-save spec after each LLM response')
    args = parser.parse_args()

    path = Path(args.spec)
    spec = load_spec(path)
    print_spec(spec, path)

    if not args.api_key:
        print("No API key — showing preset spec only.")
        print("To refine: set ANTHROPIC_API_KEY or pass --api-key sk-ant-...")
        return

    refine_loop(spec, path, args.api_key, args.save)


if __name__ == '__main__':
    main()
