# AI Core Hub

An Android app that hosts on-device AI models (text generation and embeddings) and
offers them to other apps through native Android IPC (a bound service with AIDL).

Client apps don't ship or load their own copy of a model. AI Core Hub downloads
each model once, keeps it in memory once and shares it between all connected apps.

- No Google Play Services, no analytics, no trackers.
- Network access is used only to download models.
- Builds run on GitHub Actions; every merge to `main` publishes a debug APK under
  [Releases](../../releases).
