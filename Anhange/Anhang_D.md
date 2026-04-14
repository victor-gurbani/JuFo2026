# Anhang D: Erweiterte Details zur Machine-Learning-Pipeline

Um die Transparenz und Reproduzierbarkeit der durchgeführten Klassifikationsexperimente (Random Forest und Neuronales Netz) weiter zu erhöhen, werden in diesem Anhang tiefere Einblicke in die Hyperparameter-Optimierung, die Baumstrukturen und die lokalen SHAP-Erklärungen gegeben.

## Vergleich mit neuronalen Netzen (MLP)
Parallel zum Random Forest wurde ein *Multi-Layer Perceptron* (MLP) trainiert. Dieses neuronale Netz durfte den vollen Merkmalsraum (30 Features) ohne Vorfilterung oder Komplexitätsbeschränkung nutzen. Es diente als Referenz für das theoretisch erreichbare Maximum an Genauigkeit auf dem 144-Stücke-Datensatz.

Das Ergebnis war überraschend: Der interpretierbare RF schnitt mit $70{,}1\,%$ Genauigkeit leicht besser ab als das unbeschränkte neuronale Netz ($68{,}7\,%$; vgl. \hyperref[fig:nn_comparison]{Abbildung~\ref*{fig:nn_comparison}}). Wir interpretieren diesen Befund dahingehend, dass die Kombination aus Kollinearitätsfilterung und striktem Pruning als effektive Regularisierung wirkte. Während das neuronale Netz auf dem relativ kleinen Datensatz (144 Werke) zu leichtem Overfitting neigte, zwang die interpretierbare Architektur das Modell dazu, sich auf die robustesten musikalischen Merkmale -- wie den Tonumfang und die harmonische Dichte -- zu konzentrieren (\hyperref[fig:rf_scatter]{Abbildung~\ref*{fig:rf_scatter}}).

![Vergleich der Klassifikationsleistung: Der interpretierbare Random Forest (RF) erzielt trotz seiner Komplexitätsbeschränkung eine leicht höhere Genauigkeit als das neuronale Netz (MLP). Die Filterung redundanter Merkmale fungiert hier als effiziente Regularisierung.](../figures/random_forest/nn_vs_rf_comparison.png)

## Hyperparameter-Optimierung und Pruning-Statistiken
Das Training des interpretierbaren Random Forests zielte auf eine maximale Reduktion der Komplexität ab, um "White-Box"-Eigenschaften zu bewahren, ohne die Genauigkeit drastisch zu opfern. Die Optimierung erfolgte mittels `RandomizedSearchCV` (5-Fold Cross-Validation) über folgendes Raster:

    - `n_estimators`: [100, 200, 300]
    - `max_depth`: [5, 10, 15, 20]
    - `min_samples_leaf`: [2, 3, 4]
    - `ccp_alpha`: [0.005, 0.01, 0.015, 0.02, 0.03]


Um den konkreten Effekt des *Cost-Complexity Prunings* (`ccp_alpha`) zu quantifizieren, wurde ein komplett unbeschränkter Basis-Forest mit dem optimierten, beschnittenen Forest verglichen.

    - **Unpruned Baseline**: 73,61\,% Genauigkeit, $\varnothing$ Baumtiefe: 7,90, Gesamtzahl der Knoten: 5.094
    - **Best Pruned Model** (`ccp_alpha=0.01`): 70,14\,% Genauigkeit, $\varnothing$ Baumtiefe: 6,66, Gesamtzahl der Knoten: 3.270

Der Verzicht auf nur $\sim$3,5\,% Genauigkeit führte zu einer massiven Reduzierung der Modellkomplexität um **35,8\,% weniger Entscheidungsknoten**.

## Beispielhafter Entscheidungsbaum
Ein Random Forest trifft Vorhersagen durch einen Mehrheitsentscheid vieler individueller Bäume. Um die Nachvollziehbarkeit zu demonstrieren, wurde der mathematisch akkurateste Einzelbaum aus dem 100-Bäume-Ensemble (Baum #71) extrahiert. Dieser Baum erreichte alleingestellt eine Genauigkeit von 72,4\,% auf dem Testset. Zur Gewährleistung uneingeschränkter Transparenz ("White-Box"-Paradigma) exportiert die Pipeline zudem alle 100 Entscheidungsbäume vollständig als grafische und textuelle Regelwerke zur manuellen Auditierung.

![Der akkurateste Einzelbaum des Random-Forest-Ensembles. Jeder Knoten zeigt das Entscheidungskriterium (Merkmal), die Gini-Impurity (Grad der Mischung), die verbleibende Anzahl von Stücken (Samples) und deren Verteilung auf die vier chronologischen Klassen: [Bach, Mozart, Chopin, Debussy].](../figures/random_forest/rf_sample_tree.png)

## Komponistenspezifische SHAP-Analysen
Während einzelne Entscheidungsbäume eine *Mikro-Logik* zeigen, aggregieren SHAP-Werte (SHapley Additive exPlanations) die spieltheoretischen Einflüsse aller Merkmale über das *gesamte* Ensemble. Da die Merkmale im Vorfeld mittels Spearman-Clustering von Kollinearität befreit wurden, zeigen diese Plots die isoliertesten und aussagekräftigsten Unterscheidungsmerkmale. Dass diese sich teilweise von den Top-Merkmalen der ANOVA/PCA unterscheiden, liegt methodisch in der Natur des Modells: Während die statistischen Tests lineare, isolierte Varianzen messen, erfasst der Random Forest *nicht-lineare, konditionale Interaktionen* (z.\,B. "Wenn Merkmal X niedrig ist, trennt Merkmal Y perfekt") und optimiert ausschließlich auf die Klassifikationsgrenze statt auf die globale Datenvarianz.

![SHAP-Werte für Johann Sebastian Bach. Niedrige Werte (blau) beim Tonumfang (`pitch_range_semitones`) und der Downbeat-Betonung (`downbeat_emphasis_ratio`) führen zu positiven SHAP-Werten (rechts der vertikalen Linie) und erhöhen somit die Wahrscheinlichkeit für Bach. Im Gegensatz dazu deuten hohe Werte (rot) bei Trugschlüssen (`deceptive_cadence_ratio`) und den Noten pro Taktschlag (`notes_per_beat`) stark auf Bach hin.](../figures/random_forest/rf_shap_summary_Bach.png)

![SHAP-Werte für Claude Debussy. Die wichtigsten Prädiktoren sind hier die harmonische Dichte (`harmonic_density_mean`) und das Dissonanzverhältnis (`dissonance_ratio`), deren hohe Werte stark auf Debussy hinweisen.](../figures/random_forest/rf_shap_summary_Debussy.png)



\clearpage
