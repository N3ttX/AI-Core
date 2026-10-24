---
name: android-github-workflow
description: Verwende diesen Skill, wenn in einem GitHub-Repository eine Android-App neu erstellt, geändert, repariert oder veröffentlicht wird, und bei allen Git-, Pull-Request-, Merge- und Release-Aufgaben in solchen Repositories. Regelt Branch, Pull Request, sofortigen Merge nach main, automatischen Debug-Build per GitHub Actions und Veröffentlichung der APK unter GitHub Releases, ohne eigenen Keystore.
---

# Android-Apps mit GitHub

Dieser Skill gilt für alle GitHub-Repositories mit einer Android-App. Wende den Ablauf bei jeder Änderung an, ohne dass der Nutzer die einzelnen Schritte vorgeben muss.

## Grundregeln

1. **main ist der veröffentlichte Stand.** Änderungen kommen nie direkt auf main. Einzige Ausnahme ist der allererste Commit eines leeren Repositories.
2. **Jede Änderung läuft über Branch, Pull Request und sofortigen Merge.** Es gibt keine Reviews, der Nutzer arbeitet allein. Der Merge erfolgt direkt nach dem Anlegen des PRs, nicht erst nach dem Build.
3. **Jeder Merge nach main startet einen Debug-Build auf GitHub** (GitHub Actions). Die APK wird als GitHub Release veröffentlicht. Gebaut wird nicht auf dem Rechner des Nutzers.
4. **Nur der Standard-Debug-Key.** Lege niemals Keystores, eigene signingConfigs für Release oder GitHub-Secrets für die Signierung an. Committe keine .jks- oder .keystore-Dateien.
5. **Bei fehlgeschlagenem Build reparieren.** Lies das Log, behebe den Fehler und liefere die Korrektur über einen neuen Branch und PR aus. Höchstens drei Versuche. Danach dem Nutzer Log-Auszüge und Vermutungen zeigen.
6. **Keine Geheimnisse committen.** Keine Passwörter, Tokens, API-Schlüssel und keine local.properties.

## Ersteinrichtung auf dem Rechner

Prüfe der Reihe nach und behebe, was fehlt:

1. `git --version`. Falls Git fehlt, dem Nutzer sagen, dass er es von git-scm.com installieren muss.
2. `gh --version` (GitHub CLI). Falls sie fehlt, installiere sie über den Paketmanager: macOS `brew install gh`, Windows `winget install --id GitHub.cli`, Linux über die Paketquelle auf cli.github.com.
3. `gh auth status`. Falls nicht angemeldet, bitte den Nutzer, `gh auth login` selbst im Terminal auszuführen. Den Login nicht selbst versuchen.
4. `gh auth setup-git`, damit git für Pushes dieselbe Anmeldung nutzt.
5. `git config user.name` und `git config user.email`. Falls leer, Name und E-Mail erfragen und mit `--global` setzen.

## Neues Repository anlegen

Falls die App noch kein Repository hat:

1. Name und Sichtbarkeit abfragen. Vorschlag: Projektordnername, privat.
2. `git init -b main`, Projekt anlegen (siehe unten), Pflichtdateien aus diesem Skill hinzufügen.
3. Ersten Commit direkt auf main erstellen.
4. `gh repo create <name> --private --source=. --remote=origin --push` (bei öffentlich entsprechend ändern).

## Ablauf bei jeder Änderung

1. `git checkout main`, dann `git pull --ff-only`.
2. `git checkout -b <branch>`. Branch-Name kurz, nur Kleinbuchstaben, Ziffern und Bindestriche, z. B. `feature/startbildschirm` oder `fix/absturz-beim-start`.
3. Änderungen umsetzen.
4. `git status` prüfen (keine ungewollten Dateien, siehe Pflichtdateien). Dann `git add -A` und `git commit -m "<kurze Beschreibung auf Deutsch>"`.
5. `git push -u origin <branch>`.
6. `gh pr create --base main --head <branch> --title "<Titel>" --body "<kurze Beschreibung>"`.
7. `gh pr merge <branch> --squash --delete-branch`.
8. `git checkout main`, `git pull --ff-only`, dann `git branch -D <branch>`. Fehler ignorieren, falls der Branch schon weg ist.
9. Build beobachten: `gh run list --workflow=android-debug.yml --branch=main --limit=1` liefert die Run-ID. Falls noch nichts da ist, nach ein paar Sekunden erneut abfragen. Danach `gh run watch <run-id> --exit-status`. Ein Build dauert oft mehrere Minuten. Bricht der Befehl wegen eines Zeitlimits ab, mit `gh run view <run-id>` den Status prüfen und erneut warten.
10. Ergebnis melden. Bei Erfolg den Release-Link nennen: `gh repo view --json url --jq .url`, dann `/releases/latest` anhängen. Bei Fehler die Ursache erklären und den Abschnitt "Build-Fehler" befolgen.

## Build-Fehler

- Ursache mit `gh run view <run-id> --log-failed` finden.
- Fehler im Code oder in der Konfiguration beheben und über einen neuen Branch und PR ausliefern, wie oben beschrieben.
- Liegt der Fehler im Workflow selbst (z. B. fehlender Gradle-Wrapper, falsche Java-Version), den Workflow korrigieren und dabei die Grundregeln einhalten.
- Ohne Code-Änderung neu bauen: `gh workflow run android-debug.yml --ref main`.

## Neue Android-App

- Kotlin, Jetpack Compose, Gradle mit Kotlin-DSL (`build.gradle.kts`), Projektstruktur wie in Android Studio.
- Aktuelle stabile Versionen von Android Gradle Plugin, Kotlin, Compose und Gradle verwenden. Bei Unsicherheit im Netz nachsehen statt zu raten.
- Der Gradle-Wrapper (`gradlew`, `gradlew.bat`, `gradle/wrapper/`) muss im Repository liegen. Ist Gradle lokal installiert, `gradle wrapper` ausführen. Falls nicht, den Nutzer bitten, einmal ein Projekt in Android Studio anzulegen (Empty Activity, Kotlin, Gradle Kotlin DSL) und den Ordner zu übergeben.
- `applicationId` nach dem Muster `com.<name>.<appname>`, nur Kleinbuchstaben und Ziffern.
- Keine eigene signingConfig eintragen.
- Lokale Builds sind optional und nur möglich, wenn ein Android SDK vorhanden ist. Maßgeblich ist der Build auf GitHub.

## Pflichtdateien

Jedes Android-Repository braucht:

1. `.github/workflows/android-debug.yml` mit exakt diesem Inhalt. Falls schon vorhanden und funktionierend, nicht verändern.

```yaml
name: Android Debug-Build

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: android-debug-build
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      GH_REPO: ${{ github.repository }}
    steps:
      - name: Code holen
        uses: actions/checkout@v4

      - name: Java einrichten
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '17'

      - name: Gradle einrichten
        uses: gradle/actions/setup-gradle@v4

      - name: Debug-Key stabil halten
        uses: actions/cache@v4
        with:
          path: ~/.android/debug.keystore
          key: android-debug-keystore-v1

      - name: Gradle-Wrapper ausführbar machen
        run: chmod +x ./gradlew

      - name: Debug-APK bauen
        run: ./gradlew assembleDebug --no-daemon

      - name: APK umbenennen
        run: |
          APK=$(find . -path '*/build/outputs/apk/debug/*.apk' | head -n 1)
          cp "$APK" "${{ github.event.repository.name }}-debug-${{ github.run_number }}.apk"

      - name: GitHub-Release erstellen
        run: |
          gh release create "build-${{ github.run_number }}" \
            "${{ github.event.repository.name }}-debug-${{ github.run_number }}.apk" \
            --target "$GITHUB_SHA" \
            --title "Debug-Build ${{ github.run_number }}" \
            --generate-notes
```

2. `.gitignore` mit mindestens diesem Inhalt. Falls schon vorhanden, ergänzen statt ersetzen:

```
# Android / Gradle
.gradle/
build/
local.properties
.idea/
.kotlin/
*.iml
captures/
.cxx/
.DS_Store

# Signierung und Geheimnisse
*.jks
*.keystore
*.apk
```

## Nie tun

- Nicht direkt auf main pushen (außer beim ersten Commit eines leeren Repositories).
- Kein `git push --force` auf main, kein `--no-verify`.
- Keine Keystores, keine Signing-Secrets, keine Release-Signierung.
- Keine Repositories, Branches oder Releases löschen ohne ausdrückliche Bestätigung des Nutzers.

## Kommunikation

Antworte auf Deutsch und knapp. Sag, was geändert wurde, welcher Release entstanden ist und was schiefgelaufen ist. Erkläre Git-Begriffe nur kurz, wenn sie vorkommen.
