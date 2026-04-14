# Anhang B: Reproduzierbarkeits-Leitfaden

Dieser Anhang dokumentiert die vollständige Software-Pipeline zur exakten Replikation aller Ergebnisse. Alle Befehle und Skripte sind im Projekt-Repository verfügbar.

## Pipeline-Architektur

\hyperref[fig:pipeline]{Abbildung~\ref*{fig:pipeline}} zeigt den vollständigen Datenfluss von den Rohdaten bis zur Visualisierung.

![Systemarchitektur der Analyse-Pipeline. Phase~1 filtert 254\,077 PDMX-Einträge auf 144 balancierte Klavierwerke. Phase~2 extrahiert 36 Merkmale in drei Kategorien. Phase~3 führt statistische Tests durch. Phase~4 erzeugt interaktive Visualisierungen und Evolutionsmetriken.](../figures/pipeline_architecture.png)

## Systemvoraussetzungen und Ressourcenbedarf


    - **Betriebssystem**: macOS, Linux oder Windows mit WSL
    - **Python**: Version 3.10 oder höher
    - **Speicherplatz**: 
    
        - PDMX-Datensatz: $\sim$56 GB (254\,077 MusicXML-Partituren)
        - Python Virtual Environment: $\sim$600 MB
        - Zwischenergebnisse (Features, Statistiken, Visualisierungen): $\sim$50 MB
    
    - **Rechenzeit**: 2--3 Stunden für vollständigen Durchlauf auf modernem Laptop
    - **RAM**: Mindestens 8 GB empfohlen (16 GB für große Bach-Choralkompilationen)


## Automatisierte Installation mit Quickstart-Skript

Das Projekt enthält ein vollautomatisches Bash-Skript, das Installation und Analyse in einem Durchlauf erledigt.

### Vollautomatischer Ablauf

```
./quickstart.sh
```

Das Skript führt automatisch folgende Schritte aus:

    - **Abhängigkeitsprüfung**: Validiert Python-Installation und `requirements.txt`
    - **Datensatz-Download**: Bietet interaktiven Download von Zenodo (15571083) an, falls PDMX-Verzeichnis fehlt. Unterstützt `aria2c` für parallele Downloads (16 Verbindungen) mit Fallback auf `wget`
    - **Virtual Environment**: Erstellt isolierte Python-Umgebung im `venv/`-Verzeichnis
    - **Dependency-Installation**: Installiert `music21`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`
    - **Pipeline-Ausführung**: Führt alle 10 Analyseschritte sequenziell aus (siehe unten)


**Alternative ohne Virtual Environment:**
```
./quickstart.sh --no-venv
```
Nutzt System-Python-Installation (für conda-Nutzer oder vorinstallierte Abhängigkeiten).

## Die 10-Stufen-Analysepipeline

Das Quickstart-Skript orchestriert folgende Schritte:

### Stufe 1--2: Korpus-Kuratierung und Parsing

**[1/10] Korpus-Kuratierung**
```
python3 src/corpus_curation.py --min-rating 0
```
*Ausgabe*: `data/curated/solo_piano_corpus.csv` (144~Werke), <br>`solo_piano_mxl_paths.txt`

**[2/10] Struktur-Parsing**
```
python3 src/score_parser.py \
    --output data/parsed/summaries.json
```
*Ausgabe*: JSON mit Taktzahl, Stimmenzahl, Dauern pro Partitur

### Stufe 3--5: Merkmals-Extraktion

**[3/10] Harmonische Merkmale**
```
python3 src/harmonic_features.py \
    --output-csv data/features/harmonic_features.csv
```
*Ausgabe*: 16 harmonische Deskriptoren + Boxplots in `figures/harmonic/`

**[4/10] Melodische Merkmale**
```
python3 src/melodic_features.py \
    --output-csv data/features/melodic_features.csv
```
*Ausgabe*: 11 melodische Deskriptoren + Boxplots in `figures/melodic/`

**[5/10] Rhythmische Merkmale**
```
python3 src/rhythmic_features.py \
    --output-csv data/features/rhythmic_features.csv
```
*Ausgabe*: 9 rhythmische Deskriptoren + Boxplots in `figures/rhythmic/`

### Stufe 6: Dimensionsreduktion und Embedding

**[6/10] PCA- und t-SNE-Projektionen**
```
# PCA mit 3D/2D-Visualisierung
python3 src/feature_embedding.py --method pca \
    --output figures/embeddings/pca_3d.html \
    --output-2d figures/embeddings/pca_2d.html \
    --loadings-csv data/stats/pca_loadings.csv \
    --clouds-output figures/embeddings/pca_clouds_3d.html \
    --clouds-output-2d figures/embeddings/pca_clouds_2d.html

# t-SNE mit angepassten Hyperparametern
python3 src/feature_embedding.py --method tsne \
    --perplexity 20 --tsne-composer-weight 4.0 \
    --output figures/embeddings/tsne_3d.html \
    --clouds-output figures/embeddings/tsne_clouds_3d.html
```
*Ausgabe*: Interaktive HTML-Scatter-Plots, Gaußsche Dichtewolken, PCA-Loadings

### Stufe 7--8: Statistische Signifikanz

**[7/10] ANOVA und Tukey-HSD-Tests**
```
python3 src/significance_tests.py \
    --anova-output data/stats/anova_summary.csv \
    --tukey-output data/stats/tukey_hsd.csv
```
*Ausgabe*: F-Statistiken, p-Werte, Bonferroni/FDR-Flags, paarweise Vergleiche

**[8/10] Signifikanz-Visualisierungen**
```
python3 src/significance_visualizations.py --top-n 15
```
*Ausgabe*: Bar-Charts, Heatmaps (Paar-Counts, Mean-Diff., Sym-Log) in <br>`figures/significance/`

### Stufe 9--10: Annotation und Aggregation

**[9/10] Annotierte Referenzpartituren**
```
python3 src/generate_selected_annotations.py
```
*Ausgabe*: 8 farbcodierte MusicXML-Dateien (2 pro Komponist) mit Dissonanz-Labels, Akkordsymbolen und optionalen PDF-Renderings

**[10/10] Master-Aggregation**
```
python3 src/aggregate_metrics.py
```
Validiert alle Outputs, druckt Korpus-Statistiken, verifiziert Konsistenz

## Manuelle Einzelschritte und Debugging

### Schnelltests mit `--limit`

Jedes Feature-Extraktions-Skript unterstützt partielle Läufe:
```
python3 src/harmonic_features.py --limit 5 \
    --output-csv /tmp/harmonic_test.csv
```
Prozessiert nur die ersten 5 Partituren für Smoke-Tests.

### Cached Runs mit `--features-from`

Wiederverwendung bereits berechneter Features ohne Neuberechnung:
```
python3 src/harmonic_features.py \
    --features-from data/features/harmonic_features.csv \
    --skip-plots
```
Nutzt existierende CSV, erzeugt nur Statistiken (nützlich für iterative Visualisierungen).

### Alternative Korpus-Varianten

Experimentiere mit unbalancierten oder entspannten Filtern:
```
python3 src/corpus_curation.py --min-rating 0 \
    --skip-license-filter \
    --max-per-composer 999 \
    --output-csv data/curated/unbalanced_corpus.csv
```

## Erweiterte Funktionen

### Externe Stück-Projektion

Projiziere beliebige MusicXML-Dateien in den PCA-Raum:
```
python3 src/highlight_pca_piece.py \
    --pieces path/to/score.mxl \
    --title "Joe Hisaishi - One Summer's Day" \
    --composer "Joe Hisaishi" \
    --output highlighted_projection.html
```
Unterstützt mehrere Stücke gleichzeitig mit individuellen Labels.

### MusicXML-Annotation mit MuseScore-Rendering

Erzeuge farbcodierte Analysepartituren mit automatischem PDF-Export:
```
python3 src/annotate_musicxml.py \
    path/to/input.mxl \
    path/to/output_annotated.mxl \
    --renderer-template "mscore -o {output} {input}" \
    --render-format pdf
```
Färbt Durchgänge (orange), Appoggiaturen (violett), andere Dissonanzen (rot), chromatische Harmonien (türkis); fügt römische Ziffern als Akkordsymbole ein.

## Validierung der Installation

### Erwartete Ergebnisse

Nach vollständigem Durchlauf sollten folgende Kennzahlen reproduziert werden:

    - **Korpusgröße**: Exakt 144 Partituren (36 pro Komponist)
    - **Gesamtdauer**: 71\,585 Viertelnoten (\textasciitilde13,3 Stunden bei 90 BPM)
    - **ANOVA-Treffer ($\alpha=0.05$)**: 29 Merkmale
    - **FDR-robuste Merkmale ($q<0.05$)**: 29 Merkmale
    - **PCA-Varianz (PC1--PC3)**: 48,2% (22,3% + 16,1% + 9,8%)
    - **Tukey-Unterschiede Debussy--Mozart**: 16 signifikante Merkmale
    - **Tukey-Unterschiede Bach--Mozart**: 10 signifikante Merkmale (bzw. 16 inklusive Zähl-/Größenmetriken)


### Schnelle Integritätsprüfung

```
# Test 1: Parsing-Integrität
python3 src/score_parser.py --limit 5 \
    --output /tmp/test_parse.json
# Erwartung: 5 erfolgreiche Parses

# Test 2: Feature-Extraktion
python3 src/harmonic_features.py --limit 5 \
    --output-csv /tmp/test_harmonic.csv
# Erwartung: 5 Zeilen mit 16 Harmonie-Spalten

# Test 3: Statistik-Pipeline
python3 src/significance_tests.py --alpha 0.05
# Erwartung: anova_summary.csv mit 36 Zeilen
```

## Häufige Probleme und Lösungen


    - **`music21` Parse-Fehler bei manchen Partituren** Einige MusicXML-Dateien enthalten Formatfehler oder sind Opus-Kompilationen. **Lösung**: Alle Skripte verwenden standardmäßig `--skip-errors`, um fehlerhafte Dateien zu überspringen. Für Debugging: `--no-skip-errors` aktivieren.
    
    - **Matplotlib Backend-Fehler auf Headless-Systemen** Server ohne Display benötigen nicht-interaktives Backend. **Lösung**: `export MPLBACKEND=Agg` vor Ausführung setzen.
    
    - **Speicherprobleme bei Bach-Chorälen** Einige Bach-Werke enthalten 100+ Choräle. **Lösung**: `-{`-limit} Parameter nutzen oder RAM auf min.\ 16\,GB erhöhen.
    
    - **Abweichende PCA-Ergebnisse trotz `random_state=42`** Unterschiedliche `scikit-learn`-Versionen können minimal andere Rotationen erzeugen. **Lösung**: Für exakte Replikation Versionen pinnen (z.\,B. via `pip freeze > requirements-lock.txt`) und die Analyse in einer frischen Umgebung ausführen.
    
    - **Zenodo-Download schlägt fehl** API-Rate-Limits oder Netzwerkprobleme. **Lösung**: Manueller Download von <https://zenodo.org/records/15571083>, Extraktion nach `15571083/`.


## Datenformat-Spezifikationen

### Korpus-CSV (`data/curated/solo_piano_corpus.csv`)

    - `composer_label`: Normalisiert (Bach | Mozart | Chopin | Debussy)
    - `title`: Werktitel aus PDMX-Metadaten
    - `mxl`: Relativer Pfad zu MusicXML-Datei (z.\,B.\ `0/46/Qmaug5p...mxl`)
    - `rating`: PDMX-Qualitätsbewertung (0.0--5.0)
    - `subset:*`: Boolsche Flags für PDMX-Subsets


### Feature-CSVs (`data/features/*.csv`)
Einheitliches Format für alle drei Feature-Typen:

    - `composer`: Komponistenname (für Gruppierung in ANOVA)
    - `title`: Werktitel
    - Feature-Spalten: Numerische Werte (NaN bei Extraktionsfehlern)


### ANOVA-Ergebnisse (`data/stats/anova_summary.csv`)

    - `feature`: Merkmalsname
    - `F_statistic`: ANOVA-F-Wert
    - `p_value`: Nicht-adjustierter p-Wert
    - `bonferroni_sig`: Bonferroni-Signifikanz (nach $\alpha/36$)
    - `fdr_sig`: Benjamini-Hochberg FDR-Signifikanz ($q<0.05$)


### Tukey-Ergebnisse (`data/stats/tukey_hsd.csv`)

    - `feature`: Merkmalsname
    - `group1`, `group2`: Komponistenpaar
    - `meandiff`: Mittlere Differenz
    - `lower`, `upper`: 95%-Konfidenzintervall
    - `reject`: Signifikanz (True/False)


\clearpage
