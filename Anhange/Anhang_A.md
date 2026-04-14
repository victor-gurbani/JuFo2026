# Anhang A: Vollständige Merkmalsdokumentation

Dieser Anhang dokumentiert alle 36 extrahierten Merkmale der drei Säulen. Jedes Merkmal wird durch seine musikologische Bedeutung, Berechnungsmethode und Interpretation erklärt.

## Harmonische Merkmale (16)

### Akkord-Basisdaten

    - **`chord_event_count`** Gesamtzahl der Akkordevents nach `chordify()`. Jedes Event repräsentiert eine simultane Sonorität über einen rhythmischen Abschnitt.
    
    - **`chord_quality_total`** Anzahl der Akkorde mit klassifizierbarer Qualität (Dur/Moll/vermindert/übermäßig/andere).


### Akkordqualitäts-Verteilung
Alle Prozentsätze relativ zu `chord_quality_total`:

    - **`chord_quality_major_pct`** Anteil Dur-Akkorde. Hohe Werte charakterisieren heitere, stabile Harmonik.
    
    - **`chord_quality_minor_pct`** Anteil Moll-Akkorde. Erhöhte Werte signalisieren modale Mischung oder Moll-Tonikalisierung.
    
    - **`chord_quality_diminished_pct`** Anteil verminderter Klänge. Typisch für Leitton-Akkorde (vii°) vor Kadenzen.
    
    - **`chord_quality_augmented_pct`** Anteil übermäßiger Dreiklänge. Selten; meist chromatische Durchgangsharmonien.
    
    - **`chord_quality_other_pct`** Nicht-terzenbasierte Akkorde (Quart-/Sekundschichtungen, Vorhalte, Cluster). Hohe Werte bei Debussy (z.\,B.\ *Debussy: Prelude Book 2 No 12 Feux d'artifice*: 77{,}8%).


### Textur und Konsonanz

    - **`harmonic_density_mean`** Durchschnittliche Anzahl verschiedener Tonklassen pro Akkord. Werte nahe 4 deuten auf erweiterte Harmonien hin (z.\,B.\ *Chopin: Étude Op.\,10 Nr.\,11*: 3{,}785).
    
    - **`dissonance_ratio`** Anteil dissonanter Akkorde nach `music21.isConsonant()`. Misst vertikale Spannungsfrequenz (z.\,B.\ *Debussy: Pour remercier la pluie au matin*: 0{,}845 vs.\ *Mozart: Tuba Mirum*: 0{,}000).


### Nichtakkordische Töne
Bezogen auf die melodische Hauptstimme; Nenner ist `dissonant_note_count`:

    - **`dissonant_note_count`** Anzahl detektierter nichtakkordischer Töne.
    
    - **`passing_tone_ratio`** Anteil Durchgangsnoten (schrittweise Bewegung in gleicher Richtung, Auflösung in Akkordton).
    
    - **`appoggiatura_ratio`** Anteil Appoggiaturen (Sprung zur Dissonanz auf betonter Zeit, schrittweise Auflösung).
    
    - **`other_dissonance_ratio`** Verbleibende Dissonanzen (Wechselnoten, Antizipationen, freie Dissonanzen).


### Römische Analyse

    - **`roman_chord_count`** Erfolgreich analysierte Akkorde mit römischen Ziffern.
    
    - **`deceptive_cadence_ratio`** Häufigkeit von Trugschlüssen (V--vi statt V--I).
    
    - **`modal_interchange_ratio`** Anteil modaler Mischklänge (z.\,B.\ bVI in Dur aus parallelem Moll; z.\,B.\ *Debussy: Petite suite*: 0{,}265).


## Melodische Merkmale (11)

### Registrale Merkmale

    - **`note_count`** Gesamtzahl melodischer Events (inkl.\ expandierter Akkordtöne). Beispiele: *Bach: chorale harmonisations 241--371* 30\,349 Events; *Mozart: Thema K.\,331* 55 Events.
    
    - **`pitch_range_semitones`** Ambitus in Halbtönen zwischen tiefster und höchster Note. Beispiele: *Mozart: Tuba Mirum* 22 Halbtöne; *Debussy: Feux d'artifice* 84 Halbtöne.


### Intervallik

    - **`avg_melodic_interval`** Mittlere Intervallgröße (absolut, Halbtöne) aufeinanderfolgender Noten (z.\,B.\ *Chopin: Étude Op.\,10 Nr.\,11*: 8{,}99).
    
    - **`melodic_interval_std`** Standardabweichung der Intervallgrößen. Hohe Werte zeigen Wechsel zwischen Schritten und großen Sprüngen (Chopin).
    
    - **`conjunct_motion_ratio`** Anteil schrittweiser Intervalle ($\leq$Ganzton).
    
    - **`melodic_leap_ratio`** Anteil von Sprüngen ($>$Ganzton). Komplementär zu `conjunct_motion_ratio`.


### Tonklassenverteilung

    - **`pitch_class_entropy`** Shannon-Entropie der 12-Ton-Verteilung (Basis 2). Misst chromatische Dichte (z.\,B.\ *Mozart: Thema K.\,331*: 2{,}296 vs.\ *Chopin: 24 Préludes Op.\,28 (Gesamt)*: 3{,}566).


### Zweistimmige Interaktion
Sopran (oberstes System) vs.\ Bass (unterstes System):

    - **`voice_independence_index`** Netto-Balance zwischen Gegen- und Parallelbewegung: $(\text{contrary}  - \text{parallel}) / \text{total}$ (z.\,B.\ *Chopin: Prélude Op.\,28 Nr.\,16*: $-1{,}0$; *Debussy: Le petit nègre*: 0{,}667).
    
    - **`contrary_motion_ratio`** Anteil gegenläufiger Bewegung.
    
    - **`parallel_motion_ratio`** Anteil gleichgerichteter Bewegung.
    
    - **`oblique_motion_ratio`** Anteil oblique Bewegung (eine Stimme statisch).


## Rhythmische Merkmale (9)

### Dauer-Statistik

    - **`note_event_count`** Gesamtzahl rhythmischer Events.
    
    - **`avg_note_duration`** Mittlere Notendauer (Viertelnoten-Einheiten).
    
    - **`std_note_duration`** Standardabweichung der Dauern.


### Metrische Aktivität

    - **`notes_per_beat`** Durchschnittliche Events pro Schlag (z.\,B.\ *Chopin: Étude Op.\,25 Nr.\,11 "Winter Wind*": 14{,}06 vs.\ *Chopin: Nocturne n20*: 0{,}63).
    
    - **`downbeat_emphasis_ratio`** Anteil starker Schläge (`beatStrength` $\geq 0{,}75$).
    
    - **`syncopation_ratio`** Anteil synkopischer Einträge (schwache Zeit, übergebunden über nächsten Schlag; z.\,B.\ *Debussy: Nocturne*: 0{,}211; *Bach: Chorale harmonisations (Compilation)*: 0{,}356).


### Muster-Komplexität

    - **`rhythmic_pattern_entropy`** Shannon-Entropie von Dauer-Trigrammen. Hohe Werte zeigen diverse rhythmische Zellen (z.\,B.\ *Debussy: Nocturne*: 6{,}90 vs.\ *Chopin: Étude Op.\,25 Nr.\,8*: 0{,}34).
    
    - **`micro_rhythmic_density`** Anteil 4-Noten-Fenster, in denen $\geq$3 Noten schnell sind ($\leq$Sechzehntel) (z.\,B.\ *Bach: Prelude BWV 1006*: 0{,}984).


### Polyrhythmik

    - **`cross_rhythm_ratio`** Anteil der Takte, in denen obere und untere Systeme verschiedene Dauernnenner verwenden (z.\,B.\ *Debussy: Feuilles mortes*: 1{,}0).


## Zusammenfassung der Merkmals-Interpretation

Die 36 Merkmale erfassen komplementäre Dimensionen musikalischer Gestaltung:

    - **Harmonik**: Erfasst vertikale Strukturen, Konsonanz/Dissonanz-Balance, nichtakkordische Verzierungen und tonale Funktionen. Debussys hohe `other_pct` und `modal_interchange_ratio` quantifizieren seine harmonische Innovativität.
    
    - **Melodik**: Misst registrale Spannweite, Intervallprofile und Stimmführungstypen. Chopins Brückenfunktion manifestiert sich in erhöhter `pitch_class_entropy` bei gleichzeitiger Bewahrung von `contrary_motion_ratio`-Werten nahe Mozart.
    
    - **Rhythmik**: Quantifiziert metrische Organisation, synkopische Spannung und polyrhythmische Schichtung. Debussys extreme `rhythmic_pattern_entropy` und `cross_rhythm_ratio` belegen seine rhythmische Vielfalt.


Diese vollständige Dokumentation ermöglicht die Replikation und Erweiterung der Analyse auf neue Komponisten oder Repertoires.

\clearpage
