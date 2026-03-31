Based on the instructions for the *Journal of New Music Research* and standard Taylor & Francis guidelines, here are the exact requirements to write and format your LaTeX document for submission.

### 1. Document Structure
Your document must be organized in the following specific order. 

**For a standard Research Article:**
1. **Title Page:** Includes the title, full names of all authors, affiliations (where the research was conducted - in this case none), and contact details (email - in this case victor.gurbani@gmail.com) for the corresponding author. You can also include ORCiDs (0009-0008-4571-5444) and social media handles.
2. **Abstract:** An unstructured summary of exactly or under 200 words.
3. **Keywords:** 3 to 6 keywords.
4. **Main Text:** Introduction, Materials and Methods, Results, Discussion.
5. **Acknowledgments**
6. **Declaration of Interest Statement** 7. **References**
8. **Appendices** (if appropriate)
9. **Tables:** Each on individual pages with captions.
10. **Figures:** High quality (1200 dpi for line art, 600 dpi for grayscale, 300 dpi for color). Or use SVGs or Mermaid Diagrams converted to tightly cropped PDFs
11. **Figure Captions:** Provided as a list.

### 2. Formatting and Style Guidelines
While the LaTeX template will handle most structural formatting, you must adhere to the following stylistic rules:
* **Spelling:** Use British spelling consistently, specifically the "-ise" variation (e.g., *organise*, not *organize*).
* **Quotations:** Use single quotation marks (‘text’). Use double quotation marks only for a quotation within a quotation (‘a quotation is “within” a quotation’). Long quotations should be indented as a block quote without quotation marks.
* **Units:** Use non-italicized SI units.
* **Headings:** If you need to style headings manually, T&F uses:
    * *First-level:* **Bold**, initial capital for proper nouns only.
    * *Second-level:* ***Bold italics***, initial capital for proper nouns.
    * *Third-level:* *Italics*, initial capital for proper nouns.
* **References:** Use the **T&F standard APA reference style**. If you are using BibTeX or BibLaTeX, ensure you are using an APA bibliography style package.

The template main.tex should be a pretty good starting point.

### 4. Mandatory Statements to Include
You must include these specific sections before your references:
* **Funding Details:** Use the exact phrasing required (e.g., *"This work was supported by the [Funding Agency] under Grant [number xxxx]."*) (in this case none).
* **Disclosure Statement:** Use the subheading "Disclosure of interest." If you have nothing to declare, you must explicitly state: *"The authors report there are no competing interests to declare."*
* **Ethics and Consent:** If your music research involved human participants (e.g., listening tests, surveys), include a statement in your Methods section confirming ethical approval from your Institutional Review Board and that informed consent was obtained. (in this case none) 

Create a LaTex Flag to ensure that one can compile both versions: The full version, and the anonymised version with redacted identifiers that could link back.

You may create the folder to edit the latex file as sections and including them back to the main.tex.
