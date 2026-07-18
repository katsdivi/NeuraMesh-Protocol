# A note for the judges

Thanks for taking the time to look at NeuraMesh.

**What this submission is.** The hackathon deliverable is the **Supermemory-powered
distributed memory mesh** — a layer that gives a private, on-device AI a memory
that is erasure-coded and scattered across your own devices, so that no single
device holds a complete readable copy and losing (or killing) one device does not
lose the memory. Each device runs its own local, self-hosted Supermemory instance;
memories are sealed and split into K-of-N shards distributed over NeuraMesh's
existing encrypted transport, and reconstructed only when a quorum of devices is
present. This feature — the shard codec, the sealing, the per-device Supermemory
integration, the on-device iPhone backend, the write/read/kill-a-peer paths, the
demo harness, and its tests — was built **during the hackathon**.

**What it builds on.** NeuraMesh itself (the NMP transport: Noise IK handshake,
AES-256-GCM, XOR FEC, Bonjour discovery, and cross-device LLM layer sharding) is a
**pre-existing project** I have been developing, and the memory mesh is layered on
top of it without modifying that transport/crypto core. The README and
`Docs/Memory_Mesh.md` are explicit and specific about **which code is pre-existing
NeuraMesh and which was newly built for this hackathon** — that split is honest and
intended to survive a commit-history check. The git log reflects this: the base
protocol's phases predate the event; the memory-mesh commits are new.

**It's an ongoing project.** NeuraMesh is under active development, so you may see
**commits land after the hackathon deadline** — polish, docs, the remaining iOS
on-device build step, and follow-ups such as Bonjour auto-discovery for the memory
mesh. The substantial majority of the *hackathon feature* was implemented within
the hackathon window; anything added afterward will be clearly dated in the commit
history, and I'm not backdating anything.

**Verifying it works.** Everything on the Mac side is real and was verified live,
including a mixed mesh (two Macs on Supermemory + one peer on the native on-device
store) where a peer was killed mid-demo and the memory still reconstructed from the
survivors. The full test suite runs green (474 tests, 0 failures). The one honest
gap: the iPhone *app* wiring compiles against the public API but the final
on-device build/run is a manual Xcode step on a signed device — this is stated
plainly in `Docs/Memory_Mesh_iOS.md`.

If anything here looks too polished to be hackathon work, please do check the
commit history — the honest split is the whole point.

Made with 🤍 by Divyam Kataria
