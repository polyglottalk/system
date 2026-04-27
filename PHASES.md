# PolyglotTalk — Development Phases

## Vision

PolyglotTalk is not just a desktop speech pipeline. The real target product is a **phone-to-phone speech translation app** that allows two users speaking different languages to communicate in near real time without depending entirely on cloud APIs.

The current desktop build is a **development harness** used to validate the ASR → MT → TTS cascade, debug latency, inspect transcript quality, and test model choices. It is not the final interaction model.

That distinction matters:

- On desktop, translated audio playback can feed back into the same microphone path.
- On mobile call audio stacks, this is handled much better through OS-level communication audio modes and acoustic echo cancellation.
- Therefore, the current “save TTS to WAV instead of immediately playing it” approach is a prototype workaround, not the intended end-state architecture.

---

## Guiding Principles

1. **Product first, paper second**  
   The roadmap should optimize for a working real-time communication system, not just what is easiest to publish.

2. **Desktop prototype is a staging layer**  
   The dashboard and local pipeline are tools for validation, debugging, and benchmarking.

3. **P2P is a core milestone, not optional future polish**  
   A translated calling experience requires bidirectional streaming between peers or peer-assisted endpoints.

4. **Latency matters as much as correctness**  
   A perfect translation that arrives too late fails the communication goal.

5. **Mobile constraints are real design constraints**  
   CPU, RAM, thermal throttling, battery use, and audio routing must eventually shape the architecture.

---

## Current State Summary

At present, PolyglotTalk has a functioning offline prototype pipeline:

- Microphone capture
- Automatic speech recognition
- Machine translation
- Text-to-speech synthesis
- Dashboard-based monitoring
- Audio saved to `output/chunk_NNNN.wav`

This is enough to prove the cascade works, but several issues remain:

- Overlap/repetition handling across ASR chunks is still imperfect.
- TTS playback is not yet integrated into a safe real-time duplex communication loop.
- The system is still desktop-centric rather than call-centric.
- P2P transport and synchronized two-user interaction are not yet implemented.

---

## Phase 0 — Stabilize the Core Offline Pipeline

### Goal
Make the single-device offline cascade reliable enough that later real-time and P2P work is built on solid ground.

### Scope
- Stabilize microphone capture
- Improve ASR chunking and buffering
- Reduce repeated or semantically duplicated chunk content
- Make sentence assembly robust
- Ensure MT and TTS components are modular and swappable
- Improve logging and observability in the dashboard

### Required outcomes
- Clean separation between capture, ASR, MT, TTS, and UI layers
- Repeatable end-to-end runs without crashes
- Better handling of overlap-induced duplicate meaning
- Latency breakdown per stage visible in logs/dashboard

### Exit criteria
- Pipeline runs consistently for several minutes without deadlocks or corruption
- ASR output no longer exhibits obvious repeated clause buildup in common test utterances
- TTS output is reliably generated for translated sentences

---

## Phase 1 — Fix Streaming Quality, Not Just Accuracy

### Goal
Move from “it works” to “it behaves like a conversational system.”

### Why this phase matters
The biggest current weakness is not raw model loading anymore; it is streaming behavior. Users experience the system through incremental output, not isolated benchmark scores.

### Scope
- Replace append-only sentence buffering with tail-correction logic
- Prefer later, better chunk interpretations over earlier partial fragments
- Investigate timestamp-aware ASR assembly
- Improve segmentation so partial phrases are not emitted too aggressively
- Reduce unnatural sentence concatenation artifacts before MT/TTS

### Required outcomes
- Fewer repeated meanings across adjacent chunks
- Better clause continuity
- Improved sentence flush behavior
- Cleaner translated text entering TTS

### Exit criteria
- Side-by-side transcript review shows substantial reduction in repeated clause artifacts
- End-user output sounds less “stitched together”
- Streaming transcripts are stable enough for real-time consumption

---

## Phase 2 — Real-Time Audio Playback Architecture

### Goal
Move beyond file-only output and build a real playback path suitable for conversational interaction.

### Important note
The current desktop implementation avoids auto-play because speaker output can be picked up again by the same microphone. That is a valid local workaround, but not the final architecture.

### Scope
- Add optional direct playback mode for controlled setups
- Support safe playback modes for headphones / isolated output paths
- Separate generated audio buffering from permanent file saving
- Build an internal playback queue with interruption policies
- Decide when partial translations should be spoken versus held

### Required outcomes
- TTS can be routed to a playback sink, not only saved as files
- Playback is queue-driven rather than ad hoc
- Audio artifacts and repeated speech are minimized

### Exit criteria
- Real-time translated playback works in controlled test setups
- Playback queue can handle consecutive translated segments without collapse
- WAV saving becomes a debug feature, not the primary interaction mode

---

## Phase 3 — Sender/Receiver Split

### Goal
Break the monolithic local prototype into communication-facing roles.

### Why this phase matters
A phone-call product cannot remain architecturally “one process that does everything.” The system needs clean boundaries between capturing speech, producing translated payloads, transmitting them, receiving them, and playing them back.

### Scope
Define clear modules:

- **Capture node**
  - mic input
  - local buffering
  - VAD / chunking

- **Inference node**
  - ASR
  - MT
  - TTS
  - optional incremental translation policy

- **Transport node**
  - send/receive translated payloads
  - metadata, timestamps, ordering, retries

- **Playback node**
  - audio queue
  - interrupt policy
  - synchronization with text transcript

### Required outcomes
- Clear interfaces between processing and transport
- The system can simulate “User A” and “User B” roles even on one machine
- Data contracts are defined for text/audio packets

### Exit criteria
- Local loopback test works using sender/receiver abstraction
- Each stage can be measured independently
- Transport can be developed without rewriting the pipeline core

---

## Phase 4 — P2P Transport Layer

### Goal
Implement actual peer-to-peer communication as a first-class capability.

### Why this is not optional
PolyglotTalk’s real product value comes from enabling two users to communicate directly in different languages. Without P2P or at least a call-session transport layer, the project remains a desktop translator demo.

### Scope
- Define session setup model
- Implement peer discovery/signaling strategy
- Exchange text/audio translation packets between peers
- Handle ordering, jitter, reconnects, and dropped packets
- Decide what is sent:
  - translated text only
  - synthesized audio only
  - both text and audio metadata

### Design questions to resolve
- Is transport fully peer-to-peer, or peer-assisted with a lightweight rendezvous server?
- Are translations synthesized on sender side, receiver side, or configurable?
- Does each side run full inference locally, or does each side send intermediate artifacts?

### Preferred direction
Start with a practical architecture:
- lightweight signaling server
- session establishment between peers
- bidirectional translated payload streaming
- local inference on each device wherever feasible

### Required outcomes
- Two endpoints can join the same session
- One peer’s translated output reaches the other in near real time
- Session survives short network disturbances

### Exit criteria
- Demo of two distinct endpoints exchanging translated payloads live
- Basic session controls exist: connect, disconnect, mute, language selection
- Transport metrics (delay, loss, reorder) are observable

---

## Phase 5 — Mobile Call Mode

### Goal
Transform the desktop prototype into the real intended form: a phone-based translated calling experience.

### Why this changes the architecture
Mobile platforms expose communication audio modes designed for duplex speech, routing, and acoustic echo cancellation. That makes them fundamentally better suited than the current desktop speaker/mic arrangement.

### Scope
- Build Android-first mobile client
- Use communication audio session mode
- Validate microphone capture while translated playback is active
- Test earpiece, speakerphone, wired headset, and Bluetooth routes
- Measure practical latency under mobile CPU/RAM/thermal limits
- Keep the desktop dashboard as a debugging backplane, not the primary product UI

### Required outcomes
- End-to-end translated call flow on mobile hardware
- Audio routing works without the current desktop feedback failure mode
- Mobile UI supports language pair selection, call controls, and transcript visibility

### Exit criteria
- Two phones can participate in a translated call session
- Playback and capture coexist acceptably in live testing
- The app works outside the lab, not just on a desktop dev machine

---

## Phase 6 — Low-End Device and Adaptive Runtime Strategy

### Goal
Ensure the system remains usable beyond high-end developer machines.

### Important realism
Artificially limiting RAM or CPU on a strong machine is useful for rough stress testing, but it is not a substitute for testing on real modest hardware. Real low-end devices differ in cache behavior, memory bandwidth, thermal limits, and background OS load.

### Scope
- Create deployment tiers:
  - high-end local
  - moderate local
  - constrained local
  - optional assisted mode
- Profile RAM, CPU, GPU, and stage-wise latency
- Add model-quality tiers and dynamic fallback policies
- Decide which models are feasible on mobile, laptop, and desktop classes
- Evaluate when TTS should be simplified or deferred

### Required outcomes
- Clear hardware-aware runtime modes
- Graceful degradation instead of hard failure
- Real measurements from at least one modest system and one mobile target

### Exit criteria
- Runtime can detect constraints and choose an operating profile
- Lower-resource devices still provide usable, if reduced-quality, communication
- Hardware claims are backed by actual measurements

---

## Phase 7 — Conversation Experience Layer

### Goal
Make the system feel like a real communication product instead of a chained inference demo.

### Scope
- Interrupt handling when both speakers talk
- Turn-taking cues
- Transcript alignment across both sides
- Partial-result display versus committed-result display
- Playback cancellation when newer, better translation arrives
- Bilingual transcript history
- User controls for replaying the last translated segment

### Required outcomes
- The app behaves predictably in real conversations
- Users can follow who said what, in which language, and when
- The system supports overlap and interruption gracefully

### Exit criteria
- Conversation flow remains understandable during rapid exchanges
- Transcript and playback stay synchronized enough for practical use
- User testing identifies fewer confusion points

---

## Phase 8 — Language Expansion and Model Strategy

### Goal
Move from EN↔HIN experimentation to a multilingual product strategy.

### Scope
- Add more Indian language pairs incrementally
- Evaluate ASR/MT/TTS support quality per language
- Standardize model registration and capability metadata
- Decide fallback behavior for unsupported language directions
- Track quality variation across languages rather than assuming parity

### Required outcomes
- Modular language-pack architecture
- Language-specific benchmarking and acceptance criteria
- Clear view of which language pairs are production-ready, experimental, or unsupported

### Exit criteria
- At least a small set of language pairs is supported under a consistent interface
- Language addition no longer requires architecture rewrites
- Quality gaps are documented explicitly

---

## Phase 9 — Evaluation, Benchmarking, and Publication Support

### Goal
Support journal/demo needs without letting them distort the actual product roadmap.

### Scope
- Benchmark latency by stage
- Measure ASR quality, translation quality, and TTS intelligibility
- Evaluate end-to-end conversational delay
- Compare overlap-fix strategies and model tiers
- Record reproducible experiment settings

### Important note
This phase supports publication, demos, and technical reporting, but it should not dictate the product architecture. If a publishable simplification conflicts with the real communication goal, the product goal wins.

### Required outcomes
- Reproducible benchmark suite
- Clean experiment configuration records
- Figures/tables for reports or papers when needed

### Exit criteria
- Experiments can be rerun
- Claims are grounded in repeatable measurements
- Publication artifacts can be generated without redefining the roadmap

---

## Immediate Priority Order

The practical next sequence should be:

1. **Phase 0** — stabilize pipeline
2. **Phase 1** — fix overlap / tail-correction / streaming transcript quality
3. **Phase 2** — playback architecture
4. **Phase 3** — sender/receiver split
5. **Phase 4** — P2P transport
6. **Phase 5** — mobile call mode
7. **Phase 6 onward** — optimization, language expansion, evaluation

This order matters. Jumping to mobile UI before transport boundaries are clear will create a tightly coupled mess. Delaying P2P too long will overfit the system to the wrong interaction model.

---

## Non-Goals for Now

To keep the roadmap focused, the following are explicitly not top priority right now:

- Perfect paper phrasing
- Over-polished desktop UI before transport architecture is proven
- Premature support for many languages before one or two call flows work properly
- Benchmark-heavy work that does not improve the actual communication experience
- Treating WAV-file output as a permanent user-facing interaction model

---

## Definition of Success

PolyglotTalk succeeds when:

- Two users on separate devices can communicate in different languages
- The system captures speech, translates it, and returns usable spoken output quickly enough for conversation
- The architecture can run in constrained environments with graceful degradation
- The product works as a communication system, not just as a chained offline inference demo

Until that happens, the project is still in prototype territory.