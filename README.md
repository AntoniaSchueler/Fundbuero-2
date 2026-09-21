# Fundbüro am Katharineum – Ultralytics-Version

Diese Version verwendet kein eigenes Teachable-Machine-Modell. Sie verwendet
das vortrainierte Ultralytics-Modell `yolo26n-cls.pt`. Laut Ultralytics ist
dieses Modell auf ImageNet vortrainiert und wird beim ersten Einsatz automatisch
heruntergeladen.

## Dateien

- `Pythoncode1.py` – Streamlit-App
- `requirements.txt` – Abhängigkeiten
- `runtime.txt` – Python 3.11
- `data/images/.gitkeep` – Speicherort für Bilder
- `.streamlit/secrets.toml.example` – Beispiel für den Verwaltungscode

## GitHub / Streamlit

Lade den Inhalt dieses Ordners in dein GitHub-Repository. In Streamlit als
Main file path `Pythoncode1.py` auswählen.

Es muss keine `.pt`-Datei hochgeladen werden. Ultralytics lädt `yolo26n-cls.pt`
beim ersten Modellaufruf automatisch.

## KI-Hinweis

Das Modell ist ein allgemeines ImageNet-Klassifizierungsmodell und wurde nicht
speziell für die fünf Fundbüro-Kategorien trainiert. Die App ordnet einige
allgemeine Modellbezeichnungen den Fundbüro-Kategorien zu. Daher kann die
automatische Kategorie falsch sein. Die Kategorie kann vor dem Speichern
manuell korrigiert werden.

## Verwaltungscode

Lokal ist der Fallback `sekretariat123`.

Für die öffentliche App kannst du in Streamlit Secrets setzen:

```toml
ADMIN_CODE = "DEIN_NEUER_CODE"
```

## Speicherung

Die App verwendet weiterhin SQLite und lokale Bilddateien, ohne Supabase.
Bei Streamlit Community Cloud ist lokaler Dateispeicher nicht als dauerhaft
garantiert. Für einen langfristigen Schulbetrieb sollte später ein dauerhafter
Speicher verwendet werden.

## Lizenz

Bitte beachte die aktuellen Ultralytics-Lizenzbedingungen. Die offizielle
Dokumentation nennt AGPL-3.0 und eine Enterprise-Lizenz als Optionen.
