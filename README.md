# ramer

ramer sits between the local coding/research tool you already use (claude code, cline, aider, pi, whatever) and ollama. you point your tool at ramer instead of ollama and just keep working. nothing else about your setup changes.

what it actually does: figures out how much of a model to put on your gpu vs your cpu for the context size you're running, and remembers that answer instead of making you re-tune it every time. if you've got more than one tool hitting the same gpu, it also stops them from kicking each other's models out of memory every few minutes, which is its own special kind of annoying.

i built this because i was sick of hand-tuning `num_gpu` every time I switched models or bumped the context window. figured other people running local models on a single consumer gpu are probably doing the same thing by hand, so here it is.

**the goal:** get more out of the gpu you already have without babysitting it.

## quick start

you'll need python 3.10+, ollama installed, and a discrete nvidia gpu (`nvidia-smi` has to actually find something). built and tested on windows. linux should work, same logic, nothing windows-specific in it, but i haven't personally run it there yet, so treat that as untested rather than promised. won't work on apple silicon, since the whole point of this is splitting a model across separate vram and system ram, and unified memory doesn't have that split to make.

```bash
pip install -r app/requirements.txt

python -m app.cli serve --port 11435
```

then point your tool's ollama base url at

```text
http://127.0.0.1:11435
```

instead of the usual `11434`. that's it, no changes on your tool's end. i've run this through aider and pi myself, actual coding tasks where the agent is editing files and calling tools, not just chatting.

if you want to poke at a model's placement yourself before trusting it live:

```bash
python -m app.cli doctor
python -m app.cli benchmark qwen2.5:14b-instruct --num-ctx 8192
```

there's a desktop gui too (`python -m app.gui.main`) if you'd rather not live in a terminal.

## what i've actually measured

comparing ollama's own default placement against what ramer finds, same model, same machine:

| model | ollama | ramer | gain |
|---|---:|---:|---:|
| qwen2.5 14b q4 | 12.4 tok/s | 19.6 tok/s | +61% |
| qwen2.5 32b q2 | 3.1 tok/s | 3.3 tok/s | +6.7% |
| qwen3.8 27b iq2 | 5.9 tok/s | 29.6 tok/s | +402% |

and then the part that actually matters more, real tasks, not just raw token speed:

running the same coding task through pi with the 14b model: 55.4s with ramer vs 89.2s without, both correct (checked with pytest after, not just taking the agent's word for it).

tried the same thing with a 32b q2 model and honestly, neither run finished the task at all. turns out better placement can't fix a model that just isn't up to the job, it can only make a capable model faster.

also tried a research task (search the web, read pages, answer). barely any difference with or without ramer, because almost all the time went to waiting on the network, not generating tokens.

which is basically the honest summary:

> ramer helps when local generation is actually the bottleneck. if you're mostly waiting on a website, a tool call, or a model too small for what you're asking of it, don't expect much from this.

## how it works, roughly

- tries out different gpu/cpu splits per model and per context length (context size changes how much room is left for the model itself, so a split that's right at 4k can be wrong at 32k)
- caches whatever split wins, so you're not re-probing every run
- if two splits are close enough on speed, picks the one with more headroom left rather than the fastest by a hair
- double-checks the cached split is still actually fast before trusting it, and quietly falls back to ollama's own safe default if something's regressed
- when a couple of tools are sharing one gpu, keeps requests for the same model together instead of thrashing back and forth between models

## what it doesn't do

doesn't do expert-level placement for moe models yet, that needs lower-level control than ollama's api gives me access to.

can't turn a model that's genuinely too big or too compressed into something that works. it makes a capable model faster, it doesn't make an incapable one capable.

ollama's still the thing actually running the model. ramer just sits in front of it and decides how the request gets handled.

## repo

```text
app/core/    hardware detection, placement, scheduling, the proxy itself
app/gui/     desktop interface
app/cli.py   command line entry point
app/tests/   tests
```

mostly this repo exists so people can run it on their own hardware and tell me straight whether it's actually useful. if it's not, i'd rather hear that than not.

## license

mit, see `LICENSE`.
