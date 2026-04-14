[← Zurück zur Übersicht](index.html)

# Anhang E: Deployment und Web-Infrastruktur

Die Überführung der lokalen Next.js-Datenvisualisierung in eine öffentliche Web-Anwendung erforderte die Überwindung spezifischer infrastruktureller und softwarearchitektonischer Hürden. Die Applikation (gehostet auf <https://empirical-music.victorgurbani.com>) läuft auf einem ressourcenlimitierten Oracle Linux Virtual Private Server (VPS) unter Apache 2.4 und PM2. Die Beschränkungen in Speicher und Arbeitsspeicher setzten signifikante Optimierungen des Build-Prozesses voraus.

## Limitierungen der Standalone-Kompilierung
Typischerweise verarbeitet das Next.js-Build-System (`Turbopack`) serverseitige Abhängigkeiten, indem es die im Quellcode verknüpften Ordnerstrukturen analysiert und in die endgültige Build-Datei integriert. In diesem Projekt operiert das Dashboard jedoch eng verzahnt mit dem Python-Backend und seinen Datensätzen (über 40\,GB an MusicXML-Dateien, extrahierten Features und gepickelten Modellen). Da Turbopack Aufrufe wie `path.join(repoRoot, rawPath)` für die serverseitigen API-Routen erfasste, versuchte der Compiler, diese gesamten 40\,GB als potenziell erforderliche serverseitige Fallback-Assets in den lokalen Build-Ordner (`.next/standalone`) zu kopieren. Dies stürzte in Schleifen ab (ENOSPC: No space left on device) oder überlastete den Arbeitsspeicher massiv.

Die Lösung war die Implementierung untrackbarer String-Interpolations-Polymorphismen in Next.js: Anstelle direkter standardisierter Bibliotheksaufrufe wertet das System die Suchpfade nun dynamisch über Laufzeit-Strings (`const joinPath = (base, ...parts) => [base, ...parts].join(path.sep)`) aus. Dadurch kompilierte Next.js in einen schmalen 50\,MB-Microserver, ohne die Datensatz-Binärdateien einzubetten. 

## Abruf dynamischer Cloud-Jobs
Weitere Hürden betrafen die dynamische Darstellung frisch berechneter PCA-Wolken. Im `standalone`-Modus "friert" der interne Node.js-Server von Next.js seinen statischen Datei-Router (für Assets innerhalb von `/public`) zum Zeitpunkt der Kompilierung ein. Starten Besucher der Webseite jedoch im Hintergrund einen Python-Subprozess, um eine neue Datensatzwolke zu berechnen, wurden die frisch generierten Metadaten (`.json` und `.html`) unter `public/generated/clouds/` vom Standalone-Server als 404 (Not Found) blockiert. Um dies zu umgehen, wurde in der Apache-Konfiguration ein Reverse-Proxy-Bypass integriert. Apache fängt Anfragen, die sich an die Verzeichnisse `/img` oder `/generated` richten (`Alias /generated /var/www/jufo2026/web-interface/public/generated`), explizit ab und serviert die frisch verfassten Dokumente nativ aus dem VPS-Dateisystem, während alle React- und API-spezifischen Endpunkte an den PM2-gewrappten Next.js-Dienst auf Port 3011 weitergeleitet werden. 

## Das Caching-Resolver-Dilemma\label{sec:caching}
Schlussendlich trat das Problem der Interoperabilität von Dateipfaden zwischen macOS-Entwicklungssystemen und Oracle-Linux-Hosting auf. Das Backend speicherte Caches mit absoluten Unix-Dateipfaden der MusicXML-Dateien, um Dimensionsreduktions-Metadaten effizienter heranzuziehen. Da die korrespondierenden MusicXML-Dateien aus Speicher-Gründen nicht auf den Server mitkopiert wurden (`--forceCache`-Flag überschreibt den Verzeichnis-Check), litten die Identifizierungs-Signaturen im Skript `embedding_cache.py` unter fehlschlagenden Zuordnungen. Durch den Umbau der Abgleichoperation mithilfe von `Path(mxl_abs_path).name.endswith(...)` wurde gewährleistet, dass das Dashboard auch dann voll funktionsfähig agiert und Einzelstücke fehlerfrei visualisiert rekonstruieren kann, wenn nur der Feature-Cache der Pipeline lokalisiert wird, selbst in Cross-OS-Szenarien ohne korrespondierenden Rohdatensatz.


[← Zurück zur Übersicht](index.html)

