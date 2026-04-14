[← Zurück zur Übersicht](index.html)

# Anhang F: Multicore-Caching-Infrastruktur

Um Analysen auf das vollständige PDMX-Archiv (254\,077 Partituren) zu ermöglichen, wurde ein skalierbares Caching-System entwickelt (`src/embedding_cache.py`). Dieses System unterstützt:

    - **Parallelisierte Feature-Extraktion**: Die Option `--workers N` verteilt die Berechnung auf $N$ CPU-Kerne. Bei 8 Kernen beschleunigt sich die Verarbeitung um den Faktor $\approx$6--7 (limitiert durch I/O).
    - **Unterbrechungsresistenz**: Der Fortschritt wird zeilenweise in die Cache-CSV geschrieben und in einer Sidecar-Datei (`*.done.txt`) protokolliert. Mit `--resume` kann eine unterbrochene Berechnung jederzeit fortgesetzt werden -- ideal für tagelange Läufe auf großen Korpora.
    - **Vollständiger Feature-Cache**: Neben den 3D-PCA-Koordinaten speichert das System optional alle 36 Feature-Werte pro Partitur (`--output-features-csv`). Dies ermöglicht *instant subset-specific PCA*: Für beliebige Komponisten-Kombinationen kann die PCA ohne erneutes Parsen der MusicXML-Dateien sofort neu berechnet werden.


 Die Cache-Integrität wird durch SHA256-Hashes der Eingabe-Feature-CSVs und JSON-Metadaten gewährleistet. Ein vollständiger Cache des PDMX-Korpus (alle 254\,077 Partituren $\times$ 36 Features) belegt etwa 180\,MB und ermöglicht Echtzeit-Exploration beliebiger Komponisten-Subsets über die Weboberfläche (siehe \hyperref[sec:weboberflache]{Abschnitt~\ref*{sec:weboberflache}}).


[← Zurück zur Übersicht](index.html)

