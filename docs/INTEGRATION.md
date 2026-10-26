# Using AI Core Hub from another app

AI Core Hub exposes everything through one bound service (`com.n3ttx.aicore.action.BIND`).
Add the `sdk` module to your app and use `AICoreClient`; it handles binding, reconnects and
the process boundary. All calls are `suspend` functions or `Flow`s and throw `AICoreException`
with a code from `AICoreContract.Error`.

```kotlin
val core = AICoreClient(context)
if (!core.isServiceInstalled()) { /* fall back or ask the user to install AI Core Hub */ }
val caps = core.capabilities()           // apiVersion, tasks, engines
```

## Tasks

| Task | Call | Model type |
|---|---|---|
| Chat / completion (streaming, JSON mode, GBNF) | `generate`, `complete` | `chat` (GGUF) |
| Chat with images: read text in photos, describe them (API 3) | `generate` with `ChatMessage.user(text, images)` | `chat` with the `vision` task |
| Text embeddings | `embed(texts)` | `embedding` (GGUF) |
| Image embeddings | `embedImages(images)` | `image_text` (CLIP, SigLIP; ONNX) |
| Text embeddings for image search | `embed(texts, type = IMAGE_TEXT)` | `image_text` |
| Face detection + face embeddings | `detectFaces(images)` | `face` (SCRFD + ArcFace; ONNX) |
| Any ONNX model with raw tensors | `runModel(model, component, inputs)` | any ONNX package |
| Token ids | `tokenize(text, model)` | `chat`, `image_text` |
| Installed models / catalog | `listModels`, `catalog` | – |
| Ask the user to install a model | `requestModel(id = …)` or `requestModel(packageJson = …)` | – |

Images are passed as `ImageInput`:

- `ImageInput.of(context, uri)` – AI Core Hub decodes the file itself (HEIF/AVIF/JPEG…, EXIF
  rotation applied). No storage permission is needed on its side: the file descriptor is passed.
- `ImageInput.of(bitmap)` – for thumbnails you already decoded.

Large images are downscaled before analysis (`maxDimension`, default 1600 px for faces and
3× the model's input size for image embeddings). Face boxes are returned both in pixels of the
analysed image (`Face.box`, with `FaceResult.width/height`) and normalized to 0..1
(`Face.boxNormalized`), which maps directly onto your own image.

Vectors are L2-normalized; compare them with a dot product / cosine similarity.

### Images in a chat (API 3)

Vision language models read and understand images: text on tickets, receipts and signs, or a
description of a photo. They are chat models whose `tasks` contain `AICoreContract.Task.VISION`;
`catalog()` reports the tasks of each catalog model, so `tasks` containing `VISION` lists all of them:
Qwen3.5 0.8B / 2B / 4B Vision and Gemma 4 E2B / E4B Vision (`qwen3.5-2b-vision-q4km`, about 2 GB,
is a good default).
Attach images to a message; they are shown to the model before the message's text:

```kotlin
val result = core.complete(GenerationRequest(
    messages = listOf(ChatMessage.user("What is the date on this ticket?", listOf(ImageInput.of(context, uri)))),
    model = "qwen3.5-2b-vision-q4km",
    options = GenerationOptions(temperature = 0f, grammar = myGbnf),
    maxImageDimension = 1280,   // default; larger reads smaller print but takes longer
))
```

At most 8 images per request. A request with images does not reuse the prompt cache. Models
without a vision projector answer `INVALID_REQUEST`. For forms, a GBNF grammar with fixed keys
works far better with small models than free JSON; ask the model to copy dates as printed and
convert them in your app.

### Background requests

AI Core Hub runs one request at a time. Batch jobs (indexing a gallery, scanning faces) should
pass `background = true` (`embedImages`, `detectFaces`, `embed`, `runModel`,
`GenerationRequest.background`): such requests wait while interactive requests of any app (a
chat, a search) are waiting or running, so those answer at once and the batch job continues after
them, between two of its items.

## Access

A new app has to be allowed once by the user. Launch `core.accessRequestIntent()` with
`startActivityForResult` (`RESULT_OK` = allowed), or just call the service: the first call fails
with `PENDING_APPROVAL` and AI Core Hub shows a notification.

## Models

`listModels()` returns what is installed, including `type`, `engine`, `tasks`, `components` and
`config` (JSON with model specific values such as `embedding_dim`, `image_size` or the face
`assign_threshold`). `catalog()` lists what AI Core Hub can install. If a model is missing, call
`requestModel(id = "…", reason = "…")`; the user confirms the download in AI Core Hub.

For a model outside the catalog, pass a package in the same JSON format as
[`catalog.json`](../app/src/main/assets/catalog.json):

```json
{
  "id": "my-clip", "name": "My CLIP", "type": "image_text", "engine": "onnx",
  "repo": "owner/repo",
  "files": [
    {"role": "vision", "path": "onnx/vision_model.onnx", "size": 123},
    {"role": "text", "path": "onnx/text_model.onnx", "size": 456},
    {"role": "tokenizer", "path": "tokenizer.json", "size": 789}
  ],
  "config": {"image_size": 224, "resize": "center_crop", "mean": [0.48, 0.46, 0.41], "std": [0.27, 0.26, 0.28],
             "max_text_length": 77, "embedding_dim": 512}
}
```

Roles used by the built-in tasks: `vision`, `text`, `tokenizer` (image-text), `detection`,
`recognition` (faces). A llama.cpp vision model is a package with `"engine": "llama"` and the
roles `model` (the GGUF) and `projector` (its `mmproj` GGUF). ONNX external data files (`*.onnx_data`) are listed as extra files with
any role and keep their original name.

## Lumen gallery

AI Core Hub's catalog contains Lumen's models under **the same ids** with the same files and
the same preprocessing (Lumen's tokenizer is used unchanged), so vectors match the ones Lumen
computes locally and an existing search index stays valid.

| Lumen today | With AI Core Hub |
|---|---|
| `EmbeddingEngine.embedImage(bitmap)` | `core.embedImages(listOf(ImageInput.of(bitmap)), model = aiModel.id).vectors[0]` |
| `EmbeddingEngine.embedText(query)` | `core.embed(listOf(query), model = aiModel.id, type = ModelType.IMAGE_TEXT).vectors[0]` |
| `FaceEngine.detect(bitmap, 0.55f, 20f)` + `embed(...)` | `core.detectFaces(listOf(ImageInput.of(bitmap)), model = faceModel.id, threshold = 0.55f, minFaceSize = 20f)[0]` |
| `ModelManager` download | `core.listModels()`; if missing `core.requestModel(id = aiModel.id, reason = "Photo search")` |

Ids: `mobileclip-s0`, `siglip2-b32-256`, `siglip2-b16-256`, `siglip2-l16-384`,
`siglip2-so400m-384`, `siglip2-giant-384`, `face-buffalo-s`, `face-buffalo-l`, `face-antelopev2`.

Notes for the migration:

- Pass `maxDimension = max(bitmap.width, bitmap.height)` to `detectFaces` if Lumen should get
  boxes and landmarks in exactly its own bitmap's pixels, or use `boxNormalized`.
- Batch several images per call (up to 256) during indexing to save round trips.
- Keep Lumen's local engine as a fallback while AI Core Hub is not installed
  (`isServiceInstalled()`), or drop it to save the model storage in Lumen.
