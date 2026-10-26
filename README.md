# AI Core Hub

An Android app that hosts on-device AI models and offers them to other apps
through native Android IPC (a bound service with AIDL). Client apps don't ship
or load their own copy of a model: AI Core Hub downloads each model once, keeps
it in memory once and shares it between all connected apps.

- **Text generation** (chat or raw prompt, streaming, JSON mode, GBNF grammars)
  and **text embeddings** with [llama.cpp](https://github.com/ggml-org/llama.cpp) (GGUF).
- **Vision chat**: vision language models (e.g. Qwen3.5 2B Vision) read text in photos and
  answer about images, with llama.cpp's multimodal library (mtmd).
- **Image search** (CLIP, MobileCLIP, SigLIP 2 image + text embeddings), **face detection and
  recognition** (InsightFace SCRFD + ArcFace) and **any ONNX model** with raw tensors, with
  [ONNX Runtime](https://onnxruntime.ai).
- Works with the [Lumen](https://github.com/Endermen9932/lumen) gallery: same model ids and
  identical preprocessing. See [docs/INTEGRATION.md](docs/INTEGRATION.md).
- A catalog of recommended models, plus any GGUF from Hugging Face, a direct
  URL or a local file. Client apps can ask for models; the user confirms.
- New client apps must be approved by the user once. Approvals are bound to the
  app's signing certificate.
- No Google Play Services, no analytics, no trackers. Network access is only used
  to download models and the catalog. Prompts never leave the device.
- Requires Android 16 (API 36), arm64.

Builds run on GitHub Actions; every merge to `main` publishes a debug APK under
[Releases](../../releases).

## Project layout

| Module | Contents |
|---|---|
| `app` | The AI Core Hub app: UI, model catalog and downloads, access control, `AICoreService` |
| `engine` | JNI bridge to llama.cpp (fetched at build time, pinned in `engine/src/main/cpp/CMakeLists.txt`) and the ONNX Runtime tasks (image-text, faces) |
| `sdk` | Client library for other apps: AIDL interface and `AICoreClient` |

## How it works

```
 Client app ──bind──▶ AICoreService (binder threads)
   (sdk)               │  checks the caller's approval
                       ▼
                 ModelRuntime ── one inference thread, request queue
                       │          LRU of loaded models within a memory budget,
                       ▼          idle models unloaded after N minutes
                 LlamaModel (engine, llama.cpp)
```

Requests and results are `Bundle`s, so new fields can be added without breaking
older clients. Tokens and embedding vectors stream back through one-way
callbacks. The prompt prefix of the previous request on the same model is
reused from the KV cache, so multi-turn chats only process the new messages.

## Using it from another app

The full API (images, faces, raw ONNX, model packages) is described in
[docs/INTEGRATION.md](docs/INTEGRATION.md). The basics:

1. Copy the `sdk` module into your project (or depend on it as a Gradle project)
   and add `implementation(project(":sdk"))`. The SDK's manifest already declares
   the `<queries>` entry needed to see AI Core Hub.
2. Ask the user for access once:

   ```kotlin
   val core = AICoreClient(context)
   val launcher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
       val approved = result.resultCode == Activity.RESULT_OK
   }
   launcher.launch(core.accessRequestIntent())
   ```

   Without this, the first call fails with `Error.PENDING_APPROVAL` and AI Core Hub
   shows a notification instead.
3. Generate text:

   ```kotlin
   core.generate(
       GenerationRequest(
           messages = listOf(ChatMessage.system("Be brief."), ChatMessage.user("What is Binder?")),
           options = GenerationOptions(maxTokens = 256),
       )
   ).collect { event ->
       when (event) {
           is GenerationEvent.Token -> print(event.text)
           is GenerationEvent.Completed -> println("\n${event.result.completionTokens} tokens")
       }
   }
   ```

   Or `core.complete(request)` for the whole result at once. Cancelling the
   collecting coroutine cancels the request in AI Core Hub.
4. Embed texts:

   ```kotlin
   val models = core.listModels()
   val embedder = models.models.first { it.id == models.defaultEmbeddingModel }
   val result = core.embed(listOf(embedder.queryPrefix.orEmpty() + "how do I bind a service"))
   val vector: FloatArray = result.vectors[0] // L2-normalized
   ```

5. Need a model that isn't installed? Ask for it by catalog id or by URL:

   ```kotlin
   core.requestModel(id = "qwen3-embedding-0.6b-q8", reason = "Semantic search in your notes")
   core.requestModel(
       url = "https://huggingface.co/<owner>/<repo>/resolve/main/<file>.gguf",
       name = "My model",
       type = AICoreContract.ModelType.CHAT,
   )
   ```

   The user sees the request in the Apps tab and decides whether to download it.

All calls throw `AICoreException` with a code from `AICoreContract.Error`.

## Model catalog

The recommended models are listed in [`app/src/main/assets/catalog.json`](app/src/main/assets/catalog.json).
The app ships with this file and can refresh it from GitHub (Models → Catalog →
Refresh), so the list can change without an app update.
