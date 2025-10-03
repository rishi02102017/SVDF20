# SVDF-20

## Dataset Construction Process

`SVDF.csv` is a subset of the complete collection of **17,500 songs**.  
For demonstration purposes, a sample of **5,000 songs** is shared so that readers can clearly understand the structure of the CSV file.

---

### Initial Dataset Plan
Before URL discovery, the initial plan was to include:
- **17,500 bonafide samples**  
- **17,500 deepfake versions**  
- **Total: 35,000 samples across 20 languages**  

The detailed breakdown of this design is provided in the paper.

---

### URL Discovery & Mapping
After performing URL discovery, two mapping files were generated:
- `bonafide_url_mapping.csv` (subset of 2,000 entries)  
- `deepfake_url_mapping.csv` (subset of 2,000 entries)  

These mapping files were then used with a download script to retrieve the audio files.

---

### Successful Downloads
The successfully retrieved items are recorded in:
- `bonafide_successful_downloads.csv` (subset of 2,000 entries)  
- `deepfake_successful_downloads.csv` (subset of 2,000 entries)  

---

### Subset Information
Each subset file contains representative samples demonstrating:
- **Language diversity** across the target languages  
- **Content type distribution** (film vs non-film)  
- **Temporal coverage** spanning multiple decades  
- **Quality metrics** and processing status  
- **Systematic collection methodology** with timestamps and confidence scores  

The complete dataset contains **an additional 12,500 songs/rows** with full 20-language coverage, including:  

> English, Hindi, Bengali, Spanish, Urdu, Mandarin, French, Portuguese, Russian, Tamil, Punjabi, Telugu, Kannada, Marathi, Malayalam, Indonesian, Japanese, German, Gujarati, and Arabic — together representing **70%+ of the world’s speakers**.  


## Notes
- This is a **research dataset** for **singing voice deepfake detection**.  
- The **complete dataset** will be released upon publication of the associated research paper.  
- If the paper is accepted, the complete dataset will be made publicly available through this repository along with detailed documentation and usage guidelines.  

---
