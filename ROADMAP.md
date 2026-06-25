# NatureDex AI — Feature Roadmap

This document tracks all planned features by phase.
Update this after each work session.

---

## ✅ Phase 1 — Core Prototype (DONE)
- Live webcam feed
- MobileNetV2 image classification
- Groq LLM structured entry generation (habitat, diet, conservation, NC context, fun fact)
- Confidence scores + alternative predictions
- Persistent collection saved to disk
- PyQt6 full desktop GUI
- Pokédex-style scan animation (scan line + corner brackets)
- AI wildlife educator chat tab

## ✅ Phase 2 — Collection UX (DONE)
- Search bar (live filter by name/category)
- Category dropdown filter (auto-populated from scan history)
- Delete entries (hover card → ✕ button)
- Achievement system (8 badges, count + category diversity)
- Toast notification on badge unlock
- Badges panel (🏆 button, locked/unlocked view)
- Achievement persistence (~/.naturedex_achievements.json)
- Window-sizing fix (no longer cuts off bottom on small screens)

---

## 🔜 Phase 3 — Recognition Upgrade
- Replace/supplement MobileNetV2 with wildlife-specific classifiers
  - Bird recognition (e.g. Cornell eBird / iNaturalist-trained model)
  - Plant identification model
  - Insect identification model
- Prompt-engineer Groq to re-rank ImageNet top-5 toward plausible real species
- Higher accuracy on actual animals/plants vs generic ImageNet labels

## 🔜 Phase 4 — Custom Model Training
- Pull labeled NC wildlife images from iNaturalist public API
- Fine-tune a classifier on North Carolina-specific species
  - Target: ~20-50 species to start (local birds, native plants, common insects)
- User correction system (see End-Stage Features below) feeds into retraining pipeline
- Demonstrates real ML engineering for CAC judges

## 🔜 Phase 5 — Polish & Packaging
- PyInstaller or briefcase build (judges can run it without installing Python)
- Clean install script / one-command setup
- Demo video for CAC submission
- Screenshots for submission page
- Final README polish

---

## 🌟 End-Stage Features (Target: before CAC submission)

These are planned but NOT being rushed — implement only after Phases 3-4 are solid.

### 1. Species Rarity System
- Per-species rarity rating based on geographic area
  - Pull from iNaturalist observation density data by region
  - Categories: Common / Uncommon / Rare / Very Rare / Endangered
- Display rarity badge on each entry card and in the main entry panel
- Rarity affects achievement difficulty (see Achievement Expansion below)

### 2. World Map — Discovery Pins
- Interactive map showing where each species was spotted
- Pin drops on scan location (requires location permission / GPS or IP-based)
- Pins color-coded by category or rarity
- Click a pin → loads that entry

### 3. Scientific Name Pronunciation
- Display scientific name with phonetic pronunciation guide
  - e.g. "Sialia sialis → sy-AY-lee-ah sy-AY-lis"
- Audio example button — plays a TTS or pre-recorded pronunciation
- Could use a TTS API (ElevenLabs, OpenAI TTS, or system TTS) for the audio

### 4. Achievement Expansion + Progress Bars
- Replace simple locked/unlocked badges with progress bars showing how close you are
  - e.g. "Discover 10 species: ██████░░░░ 6/10"
- More achievement categories:
  - Rarity-based: "Find an Endangered species", "Find a Critically Endangered species"
  - Location-based: "Discover species in 3 different states"
  - Category sweeps: "Find a bird, plant, insect, AND reptile"
  - Streak: "Scan 5 days in a row"
  - NC-specific: "Find 5 species native to North Carolina"
- Visual polish: animated progress bar fill on unlock

### 5. User Correction / Model Feedback System
- "Wrong answer? Tell us" button on every entry
- User can type the correct species name
- Submission is logged locally to a corrections.json file
- When custom model (Phase 4) exists:
  - Corrections feed into a retraining queue
  - Periodically retrain/fine-tune the model with accumulated corrections
  - Crowdsourced improvement over time
- For CAC: even without live retraining, the feedback UI + pipeline design is impressive

---

## Technical Debt / Known Issues
- [ ] Ask AI tab silently does nothing if no scan has been done yet — needs visible error state
- [ ] ChatWorker exceptions not surfaced to UI — errors swallowed silently
- [ ] Camera index hardcoded to 0 — no fallback if camera 0 unavailable
- [ ] No offline fallback entry if Groq API is down during a demo
