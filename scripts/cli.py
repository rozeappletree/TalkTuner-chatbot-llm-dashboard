#!/usr/bin/env python3
"""
talktuner_cli.py -- a minimal, dependency-light reimplementation of the TalkTuner
backend from "Designing a Dashboard for Transparency and Control of Conversational
AI" (Chen et al., arXiv:2406.07882).

It does the two things the paper's interface does, without the interface:

  READ    run the released *reading* probes on the residual stream and print the
          chatbot's internal model of you (age / gender / education / socioeco)

  CONTROL steer generation by translating the residual stream along the released
          *control* probe weights, i.e. pin an attribute and watch the answer move

Everything needed is vendored here so you do not have to fight the repo's
inconsistent `from dataset import ...` vs `from src.dataset import ...` imports.
You only need the checkpoint archives from the repo.

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------
  git clone https://github.com/yc015/TalkTuner-chatbot-llm-dashboard.git
  cd TalkTuner-chatbot-llm-dashboard/data/probe_checkpoints
  unzip reading_probe.zip && unzip controlling_probe.zip

  pip install torch transformers accelerate sentencepiece protobuf
  huggingface-cli login          # Llama-2 is gated; request access first

  python talktuner_cli.py --probe-dir /path/to/data/probe_checkpoints

--------------------------------------------------------------------------------
HARDWARE
--------------------------------------------------------------------------------
Llama-2-13b-chat in fp16 is ~26 GB of weights. A 40 GB card is comfortable.
On 24 GB, pass --load-8bit (needs bitsandbytes). Be aware that quantization
perturbs the activations the probes were fit on, so readings get noisier and
steering strengths may need retuning. fp16 is the faithful configuration.

--------------------------------------------------------------------------------
COMMANDS (inside the REPL)
--------------------------------------------------------------------------------
  /pin gender female        pin an attribute, steering every subsequent response
  /pin socioeco low
  /unpin gender             remove one pin;  /unpin        removes all
  /strength 12              intervention strength N (paper uses 8)
  /layers 19 29             half-open decoder-layer range to steer (paper: 19-29)
  /readlayer 30             which layer the reading probes are evaluated at
  /regen                    re-answer the last user message under current pins
  /compare gender           answer the last message twice, once per extreme
  /dash                     reprint the dashboard
  /reset                    clear conversation history
  /quit
"""


"""LIMITATIONS (ai generated)

High
Never run end to end. No GPU or Llama-2 access in my sandbox. Prompt formatting, checkpoint loading, and the steering math are unit-tested against the repo; the actual model.generate path with real hooks is unverified. Budget an hour for first-run friction.
Transformers version sensitivity. The repo pins transformers==4.45.1, where LlamaDecoderLayer returns a tuple. Some newer refactors return a bare tensor, and blindly indexing output[0] would have steered along the batch axis. I now handle both and raise a loud error on anything else, but I can't test against every version. If you hit trouble, pin 4.45.1.
--load-8bit / --load-4bit are not faithful. The probes were fit on fp16 activations. Quantization shifts those activations, so readings get noisier and steering may need a higher N. Fine for a smoke test, not for anything you'd report. fp16 is the real configuration.

Medium
hidden_states[30] for reading is my choice, not the paper's. The paper never names a dashboard layer; I took 30 from Appendix B's ablation, while Figure 23 uses 26. Sweep it with /readlayer before trusting any specific number.
Steering direction is the raw probe weight row, not a contrastive difference. Adding N·w_female is what the repo does, so this is faithful — but it isn't a clean "gender axis," and the probe bias is ignored entirely. Fine for reproducing their outputs, questionable if you want to make claims about the geometry.
Multi-attribute pinning goes beyond the paper. Summing vectors is my extension; the repo steers one attribute at a time, and the paper's limitations section explicitly flags that attribute independence is an untested assumption. Interactions are unvalidated.
KV cache staleness, inherited from the repo. The hook fires on layer L's output, so layer L's own keys and values for that position were already computed from unsteered input. The intervention propagates from L+1 onward. Same as the paper's implementation, but it means "steer layers 19–29" is slightly looser than it sounds.
Four extra full forward passes per turn — one per attribute, each over the whole conversation, on top of generation. Noticeably slow on long chats. Batching the four reading prompts with left padding would mostly fix it; I didn't.
Truncation past 2048 tokens. The probes were trained with a 2048 cap. I switched the reading path to left truncation, since right truncation would silently amputate the I think the {attribute} of this user is suffix and read a garbage token — the repo's TextDataset has exactly that bug. But left truncation still drops the start of a long conversation, and readings past that point drift from training conditions.

Low
Won't reproduce Table 2. No batch mode, no question-file runner, no GPT-4 judge. This is an interactive probe, not an eval harness. The four causality notebooks remain the path for that number.
No reading-probe steering arm. Only control probes steer, so the comparison the paper actually rests on in Section 5.1 isn't available here. Adding it means rescaling the translation to equal L2 distance.
The "unknown" threshold is invented. The paper's dashboard shows unknown early in a conversation; the mechanism isn't documented, so I used a crude 0.5 sigmoid cutoff. Treat it as cosmetic.
Dashboard reads pre-steering by design. It shows what the model infers, not what it believes under your pins. Defensible, but it's a choice, not a neutral one.
Greedy decoding only, matching the paper. No sampling exposed.
No context-length management. Long conversations will eventually hit the 3500-token generation cap with no graceful handling.
/compare mutates history by popping trailing assistant turns before regenerating.

One deliberate deviation worth flagging: the repo calls model.resize_token_embeddings to add a <pad> token, which appends a randomly-initialized embedding row. Harmless for their batched generation, unnecessary at batch size 1, and I skip it. If you ever diff my outputs against their published intervention files and see drift, that's one candidate.
"""

import argparse
import os
import re
import sys
import textwrap

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Attribute schema. Class orders are taken verbatim from the label_to_id dicts in
# notebooks/train_probes/train_read_and_controlling_probes.ipynb -- getting these
# wrong silently steers toward the wrong class, so do not reorder them.
# ------------------------------------------------------------------------------

ATTRIBUTES = {
    "age": {
        "classes": ["child", "adolescent", "adult", "older adult"],
        "read_phrase": "age",
    },
    "gender": {
        "classes": ["male", "female"],
        "read_phrase": "gender",
    },
    "education": {
        "classes": ["someschool", "highschool", "collegemore"],
        "read_phrase": "education level",
    },
    "socioeco": {
        "classes": ["low", "middle", "high"],
        "read_phrase": "socioeconomic status",
    },
}

PRETTY = {
    "someschool": "some schooling",
    "highschool": "high school",
    "collegemore": "college & beyond",
    "low": "lower",
    "middle": "middle",
    "high": "upper",
}

HIDDEN_DIM = 5120          # Llama-2-13b residual width
N_HIDDEN_STATES = 41       # embeddings + 40 decoder layers

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Please ensure that your responses "
    "are socially unbiased and positive in nature. If a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please "
    "don't share false information."
)


# ------------------------------------------------------------------------------
# Probe definition. Must match src/probes.py:LinearProbeClassification with
# logistic=True, or the state dicts will not load.
# ------------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, n_classes), nn.Sigmoid())

    def forward(self, act):
        return self.proj(act)

    @property
    def weight(self):
        return self.proj[0].weight          # [n_classes, input_dim]


CKPT_RE = re.compile(r"^(?P<attr>[a-z]+)_probe_at_layer_(?P<layer>\d+)\.pth$")


def load_probe_bank(directory, device):
    """directory -> {attribute: {layer_index: LinearProbe}}

    Layer indices are into hidden_states, so index 0 is the embedding output and
    index k is the output of decoder layer k-1.
    """
    if not os.path.isdir(directory):
        sys.exit(f"[fatal] not a directory: {directory}\n"
                 f"        did you unzip reading_probe.zip / controlling_probe.zip?")

    bank = {}
    for fname in os.listdir(directory):
        m = CKPT_RE.match(fname)
        if not m:
            continue                        # skips the *_final.pth variants
        attr, layer = m.group("attr"), int(m.group("layer"))
        if attr not in ATTRIBUTES:
            continue
        probe = LinearProbe(HIDDEN_DIM, len(ATTRIBUTES[attr]["classes"]))
        state = torch.load(os.path.join(directory, fname), map_location="cpu")
        probe.load_state_dict(state)
        probe.to(device).eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        bank.setdefault(attr, {})[layer] = probe

    if not bank:
        sys.exit(f"[fatal] no probe checkpoints found in {directory}")

    missing = [a for a in ATTRIBUTES if a not in bank]
    if missing:
        print(f"[warn] no checkpoints for: {', '.join(missing)}")
    for attr, layers in sorted(bank.items()):
        if len(layers) != N_HIDDEN_STATES:
            print(f"[warn] {attr}: found {len(layers)}/{N_HIDDEN_STATES} layers")
    return bank


# ------------------------------------------------------------------------------
# Llama-2 chat formatting, ported from src/dataset.py:llama_v2_prompt so that the
# probes see exactly the token stream they were trained on.
# ------------------------------------------------------------------------------

def llama_v2_prompt(messages, system_prompt=DEFAULT_SYSTEM_PROMPT):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    msgs = [{"role": "system", "content": system_prompt}] + list(messages)
    msgs = [{
        "role": msgs[1]["role"],
        "content": B_SYS + msgs[0]["content"] + E_SYS + msgs[1]["content"],
    }] + msgs[2:]

    out = [
        f"{BOS}{B_INST} {p['content'].strip()} {E_INST} {a['content'].strip()} {EOS}"
        for p, a in zip(msgs[::2], msgs[1::2])
    ]
    if msgs[-1]["role"] == "user":
        out.append(f"{BOS}{B_INST} {msgs[-1]['content'].strip()} {E_INST}")
    return "".join(out)


def reading_prompt(messages, attribute, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """The exact input the reading probes were trained on.

    A conversation ending in a user turn, plus a forced assistant prefix
    'I think the {attribute} of this user is'. The probe reads the last token.
    The leading <s> is stripped because the tokenizer re-adds BOS -- this
    mirrors the new_format branch of src/dataset.py.
    """
    text = llama_v2_prompt(messages, system_prompt)
    if text.startswith("<s>"):
        text = text[len("<s>"):]
    phrase = ATTRIBUTES[attribute]["read_phrase"]
    return text + f" I think the {phrase} of this user is"


# ------------------------------------------------------------------------------
# Steering. Ported from src/intervention_utils.py:edit_inter_rep_multi_layers.
#
# The intervention is  h <- h + N * w_c  applied to the LAST token position at
# every forward pass, for each decoder layer in the chosen range. Because it
# fires on every pass, it applies during prefill and again for each decoded
# token, which is what the paper means by "applied repeatedly on the last input
# token representation until the response was complete."
# ------------------------------------------------------------------------------

class Steerer:
    def __init__(self, model, control_bank, device):
        self.model = model
        self.bank = control_bank
        self.device = device
        self.pins = {}                       # attribute -> class name
        self.strength = 8.0                  # paper's N
        self.lo, self.hi = 19, 29            # half-open, paper's "20th to 29th"
        self.read_layer = 30
        self._vectors = {}                   # decoder layer index -> [D] tensor
        self._handles = []
        self._install()

    # -- vector bookkeeping ----------------------------------------------------

    def _rebuild(self):
        self._vectors = {}
        if not self.pins:
            return
        for layer in range(self.lo, self.hi):
            acc = torch.zeros(HIDDEN_DIM, dtype=torch.float32, device=self.device)
            for attr, cls in self.pins.items():
                # decoder layer L produces hidden_states[L + 1]
                probe = self.bank[attr][layer + 1]
                idx = ATTRIBUTES[attr]["classes"].index(cls)
                acc += probe.weight[idx].to(torch.float32)
            self._vectors[layer] = acc

    def pin(self, attr, cls):
        self.pins[attr] = cls
        self._rebuild()

    def unpin(self, attr=None):
        if attr is None:
            self.pins.clear()
        else:
            self.pins.pop(attr, None)
        self._rebuild()

    def set_strength(self, n):
        self.strength = float(n)
        self._rebuild()

    def set_layers(self, lo, hi):
        self.lo, self.hi = int(lo), int(hi)
        self._uninstall()
        self._install()
        self._rebuild()

    @property
    def active(self):
        return bool(self.pins)

    # -- hooks -----------------------------------------------------------------

    def _install(self):
        layers = self.model.model.layers
        for i in range(self.lo, min(self.hi, len(layers))):
            self._handles.append(layers[i].register_forward_hook(self._make_hook(i)))

    def _uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, layer_idx):
        def hook(module, args, output):
            if not self.pins:
                return output
            vec = self._vectors.get(layer_idx)
            if vec is None:
                return output

            # transformers <=4.4x returns (hidden, ...); some newer refactors of
            # LlamaDecoderLayer return a bare tensor. Handle both rather than
            # indexing [0] and silently steering the wrong axis.
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if not torch.is_tensor(hidden) or hidden.dim() != 3:
                raise RuntimeError(
                    f"unexpected decoder layer output at layer {layer_idx}: "
                    f"{type(output)}. This script assumes a [B, T, D] hidden state; "
                    f"check your transformers version (repo pins 4.45.1)."
                )

            # .to(device) matters under device_map='auto': the probe bank lives on
            # one device while layer L may be sharded onto another.
            delta = (vec * self.strength).to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, -1] = hidden[:, -1] + delta
            return (hidden,) + tuple(output[1:]) if is_tuple else hidden
        return hook


# ------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------

@torch.no_grad()
def read_user_model(model, tokenizer, reading_bank, messages, layer, device):
    """Returns {attribute: [(class, score), ...]} sorted by score, descending."""
    result = {}
    for attr in ATTRIBUTES:
        if attr not in reading_bank or layer not in reading_bank[attr]:
            continue
        text = reading_prompt(messages, attr)
        # LEFT truncation: the probe reads the final token, so right-truncating a
        # long conversation would silently amputate the
        # "I think the {attribute} of this user is" suffix and read garbage.
        # (The repo's TextDataset right-truncates and has this bug.)
        prev_side = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = "left"
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        tokenizer.truncation_side = prev_side

        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, return_dict=True)
        act = out.hidden_states[layer][0, -1]
        probe = reading_bank[attr][layer]
        act = act.to(device=probe.weight.device, dtype=torch.float32)
        scores = probe(act).tolist()
        pairs = list(zip(ATTRIBUTES[attr]["classes"], scores))
        result[attr] = sorted(pairs, key=lambda p: -p[1])
    return result


def render_dashboard(reading, pins, unknown_threshold=0.5, width=22):
    lines = ["", "  ┌─ internal user model " + "─" * (width + 22) + "┐"]
    for attr in ATTRIBUTES:
        if attr not in reading:
            continue
        top_cls, top_score = reading[attr][0]
        pinned = pins.get(attr)
        label = PRETTY.get(top_cls, top_cls)
        if top_score < unknown_threshold:
            shown, conf = "unknown", ""
        else:
            shown, conf = label, f"{top_score * 100:5.1f}%"
        bar = "█" * int(round(top_score * width)) + "·" * (width - int(round(top_score * width)))
        flag = f"  << PINNED {PRETTY.get(pinned, pinned)}" if pinned else ""
        lines.append(f"  │ {attr:<10} {shown:<16} {conf:>6}  {bar}{flag}")
    lines.append("  └" + "─" * (width + 45) + "┘")

    detail = []
    for attr in ATTRIBUTES:
        if attr not in reading:
            continue
        inner = "  ".join(f"{PRETTY.get(c, c)} {s * 100:.0f}" for c, s in reading[attr])
        detail.append(f"    {attr:<10} {inner}")
    lines.append("  raw probe scores (sigmoid, one-vs-rest -- they need not sum to 100):")
    lines.extend(detail)
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, messages, device, max_new_tokens=384, strip_bos=False):
    text = llama_v2_prompt(messages)
    if strip_bos and text.startswith("<s>"):
        text = text[len("<s>"):]
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                    # paper uses greedy for reproducibility
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def wrap(s, indent="  "):
    return "\n".join(
        textwrap.fill(line, 88, initial_indent=indent, subsequent_indent=indent) or indent
        for line in s.split("\n")
    )


# ------------------------------------------------------------------------------
# REPL
# ------------------------------------------------------------------------------

BANNER = """
  TalkTuner probe console -- Llama-2-13b-chat internal user model
  reading probes evaluated at layer {rl}; steering layers [{lo},{hi}) at N={n}
  /help for commands, /quit to exit
"""

HELP = """
  /pin <attr> <class>    pin an attribute and steer every response
  /unpin [attr]          drop one pin, or all pins
  /strength <float>      intervention strength N (paper: 8)
  /layers <lo> <hi>      half-open decoder layer range (paper: 19 29)
  /readlayer <int>       hidden_states index the reading probes use
  /regen                 re-answer the last user message under current pins
  /compare <attr>        answer last message once per extreme class of <attr>
  /dash                  reprint the dashboard
  /reset                 clear the conversation
  /quit

  attributes and classes:
    age        child | adolescent | adult | older adult
    gender     male | female
    education  someschool | highschool | collegemore
    socioeco   low | middle | high
"""


def main():
    ap = argparse.ArgumentParser(description="Read and steer Llama-2-13b-chat's internal user model.")
    ap.add_argument("--probe-dir", required=True,
                    help="directory containing unzipped reading_probe/ and controlling_probe/")
    ap.add_argument("--model", default="NousResearch/Llama-2-13b-chat-hf")
    ap.add_argument("--read-layer", type=int, default=30)
    ap.add_argument("--strength", type=float, default=8.0)
    ap.add_argument("--layers", type=int, nargs=2, default=[19, 29], metavar=("LO", "HI"))
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--unknown-threshold", type=float, default=0.5)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--strip-bos", action="store_true",
                    help="strip the leading <s> before generation; the repo does not, "
                         "so leave this off to match their published outputs")
    ap.add_argument("--demo", metavar="ATTR",
                    help="non-interactive: answer --demo-question under both extremes of ATTR")
    ap.add_argument("--demo-question", default=None)
    args = ap.parse_args()

    read_dir = os.path.join(args.probe_dir, "reading_probe")
    ctrl_dir = os.path.join(args.probe_dir, "controlling_probe")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device visible; this will be extremely slow.")

    print("[..] loading probes")
    reading_bank = load_probe_bank(read_dir, device)
    control_bank = load_probe_bank(ctrl_dir, device)

    print(f"[..] loading {args.model} (this takes a few minutes on first run)")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if args.load_8bit:
        kwargs = {"load_in_8bit": True, "device_map": "auto"}
    elif args.load_4bit:
        kwargs = {"load_in_4bit": True, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    if args.load_8bit or args.load_4bit:
        print("[warn] quantized weights shift the activations the probes were fit on; "
              "expect noisier readings and retune --strength if steering looks weak.")

    steerer = Steerer(model, control_bank, device)
    steerer.set_strength(args.strength)
    steerer.set_layers(*args.layers)
    steerer.read_layer = args.read_layer

    # -- non-interactive comparison ------------------------------------------

    if args.demo:
        attr = args.demo
        if attr not in ATTRIBUTES:
            sys.exit(f"[fatal] unknown attribute {attr}")
        q = args.demo_question or {
            "gender": "What should I wear on a first date? Please give me at least three specific outfit suggestions!",
            "socioeco": "Hi! I am going to Hawaii this summer! What would be the best transportation method for me to get there? I currently live in Los Angeles, CA.",
            "age": "Can you suggest some fun outdoor activities for me?",
            "education": "Explain the central dogma of molecular biology.",
        }[attr]
        classes = ATTRIBUTES[attr]["classes"]
        print(f"\n  question: {q}\n")
        for cls in (classes[0], classes[-1]):
            steerer.unpin()
            steerer.pin(attr, cls)
            ans = generate(model, tokenizer, [{"role": "user", "content": q}],
                           device, args.max_new_tokens, args.strip_bos)
            print("=" * 92)
            print(f"  pinned {attr} = {PRETTY.get(cls, cls)}   (N={steerer.strength}, "
                  f"layers [{steerer.lo},{steerer.hi}))")
            print("=" * 92)
            print(wrap(ans))
            print()
        return

    # -- interactive ----------------------------------------------------------

    print(BANNER.format(rl=steerer.read_layer, lo=steerer.lo,
                        hi=steerer.hi, n=steerer.strength))
    messages = []
    last_reading = {}

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0]

            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/reset":
                messages, last_reading = [], {}
                print("  conversation cleared")
            elif cmd == "/dash":
                print(render_dashboard(last_reading, steerer.pins, args.unknown_threshold)
                      if last_reading else "  nothing read yet")
            elif cmd == "/pin" and len(parts) >= 3:
                attr = parts[1]
                cls = " ".join(parts[2:])
                if attr not in ATTRIBUTES:
                    print(f"  unknown attribute {attr}")
                elif cls not in ATTRIBUTES[attr]["classes"]:
                    print(f"  {attr} classes: {', '.join(ATTRIBUTES[attr]['classes'])}")
                else:
                    steerer.pin(attr, cls)
                    print(f"  pinned {attr} = {cls}  (N={steerer.strength})")
            elif cmd == "/unpin":
                steerer.unpin(parts[1] if len(parts) > 1 else None)
                print(f"  pins: {steerer.pins or 'none'}")
            elif cmd == "/strength" and len(parts) == 2:
                steerer.set_strength(float(parts[1]))
                print(f"  N = {steerer.strength}")
            elif cmd == "/layers" and len(parts) == 3:
                steerer.set_layers(parts[1], parts[2])
                print(f"  steering layers [{steerer.lo},{steerer.hi})")
            elif cmd == "/readlayer" and len(parts) == 2:
                steerer.read_layer = int(parts[1])
                print(f"  reading at hidden_states[{steerer.read_layer}]")
            elif cmd in ("/regen", "/compare"):
                if not messages:
                    print("  no message to answer yet")
                    continue
                while messages and messages[-1]["role"] == "assistant":
                    messages.pop()
                if cmd == "/regen":
                    ans = generate(model, tokenizer, messages, device,
                                   args.max_new_tokens, args.strip_bos)
                    print(f"\nbot >\n{wrap(ans)}\n")
                    messages.append({"role": "assistant", "content": ans})
                else:
                    attr = parts[1] if len(parts) > 1 else None
                    if attr not in ATTRIBUTES:
                        print(f"  usage: /compare <{'|'.join(ATTRIBUTES)}>")
                        continue
                    saved = dict(steerer.pins)
                    for cls in (ATTRIBUTES[attr]["classes"][0], ATTRIBUTES[attr]["classes"][-1]):
                        steerer.unpin()
                        steerer.pin(attr, cls)
                        ans = generate(model, tokenizer, messages, device,
                                       args.max_new_tokens, args.strip_bos)
                        print(f"\n--- {attr} pinned to {PRETTY.get(cls, cls)} ---")
                        print(wrap(ans))
                    steerer.unpin()
                    for a, c in saved.items():
                        steerer.pin(a, c)
                    print()
            else:
                print("  unrecognized; /help")
            continue

        # normal turn
        messages.append({"role": "user", "content": line})

        # read BEFORE steering, so the dashboard shows what the model actually
        # infers rather than what you told it to believe
        was = dict(steerer.pins)
        steerer.unpin()
        last_reading = read_user_model(model, tokenizer, reading_bank, messages,
                                       steerer.read_layer, device)
        for a, c in was.items():
            steerer.pin(a, c)

        print(render_dashboard(last_reading, steerer.pins, args.unknown_threshold))

        ans = generate(model, tokenizer, messages, device,
                       args.max_new_tokens, args.strip_bos)
        print(f"bot >\n{wrap(ans)}\n")
        messages.append({"role": "assistant", "content": ans})


if __name__ == "__main__":
    main()
