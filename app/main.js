const state = {
  textData: null,
  bdbData: null,
  selectedRef: { book: null, chapter: null, verse: null },
  selectedWordIndex: null,
};

const els = {
  bookSelect: document.getElementById("bookSelect"),
  chapterSelect: document.getElementById("chapterSelect"),
  verseSelect: document.getElementById("verseSelect"),
  hebrewLine: document.getElementById("hebrewLine"),
  emptyState: document.getElementById("emptyState"),
  wordDetail: document.getElementById("wordDetail"),
  detailHebrew: document.getElementById("detailHebrew"),
  detailLemma: document.getElementById("detailLemma"),
  detailMorph: document.getElementById("detailMorph"),
  detailBdb: document.getElementById("detailBdb"),
  annotationInput: document.getElementById("annotationInput"),
  saveAnnotationBtn: document.getElementById("saveAnnotationBtn"),
  bdbEntry: document.getElementById("bdbEntry"),
};

init().catch((err) => {
  console.error(err);
  alert("Failed to load data files. Check /data/*.json");
});

async function init() {
  const [textData, bdbData] = await Promise.all([
    loadJsonWithFallback("/data/wlc_full.json", "/data/wlc_sample.json"),
    loadJsonWithFallback("/data/bdb_full.json", "/data/bdb_sample.json"),
  ]);

  state.textData = textData;
  state.bdbData = bdbData;

  buildBookSelect();
  wireEvents();
}

async function loadJsonWithFallback(primary, fallback) {
  const primaryResponse = await fetch(primary);
  if (primaryResponse.ok) {
    return primaryResponse.json();
  }
  const fallbackResponse = await fetch(fallback);
  if (!fallbackResponse.ok) {
    throw new Error(`Failed to load ${primary} and fallback ${fallback}`);
  }
  return fallbackResponse.json();
}

function wireEvents() {
  els.bookSelect.addEventListener("change", () => {
    state.selectedRef.book = els.bookSelect.value;
    buildChapterSelect();
  });

  els.chapterSelect.addEventListener("change", () => {
    state.selectedRef.chapter = els.chapterSelect.value;
    buildVerseSelect();
  });

  els.verseSelect.addEventListener("change", () => {
    state.selectedRef.verse = els.verseSelect.value;
    state.selectedWordIndex = null;
    renderVerse();
    clearWordDetail();
  });

  els.saveAnnotationBtn.addEventListener("click", saveAnnotation);
}

function buildBookSelect() {
  const books = Object.keys(state.textData.books);
  fillSelect(els.bookSelect, books);
  state.selectedRef.book = books[0];
  buildChapterSelect();
}

function buildChapterSelect() {
  const chapters = Object.keys(state.textData.books[state.selectedRef.book].chapters);
  fillSelect(els.chapterSelect, chapters);
  state.selectedRef.chapter = chapters[0];
  buildVerseSelect();
}

function buildVerseSelect() {
  const verses = Object.keys(
    state.textData.books[state.selectedRef.book].chapters[state.selectedRef.chapter].verses,
  );
  fillSelect(els.verseSelect, verses);
  state.selectedRef.verse = verses[0];
  renderVerse();
  clearWordDetail();
}

function fillSelect(select, values) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

function currentWords() {
  return state.textData.books[state.selectedRef.book].chapters[state.selectedRef.chapter].verses[
    state.selectedRef.verse
  ].words;
}

function renderVerse() {
  const words = currentWords();
  els.hebrewLine.innerHTML = "";

  words.forEach((word, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "word";
    btn.dataset.index = String(index);

    const heb = document.createElement("span");
    heb.textContent = word.text;
    btn.appendChild(heb);

    const annotation = readAnnotation(state.selectedRef, index);
    if (annotation) {
      const gloss = document.createElement("span");
      gloss.className = "annotation";
      gloss.textContent = annotation;
      btn.appendChild(gloss);
    }

    btn.addEventListener("click", () => selectWord(index));
    els.hebrewLine.appendChild(btn);
  });
}

function selectWord(index) {
  state.selectedWordIndex = index;
  const words = currentWords();
  const word = words[index];

  [...els.hebrewLine.querySelectorAll(".word")].forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.index) === index);
  });

  els.emptyState.classList.add("hidden");
  els.wordDetail.classList.remove("hidden");

  els.detailHebrew.textContent = word.text;
  els.detailLemma.textContent = word.lemma;
  els.detailMorph.textContent = word.morph || "-";
  els.detailBdb.textContent = word.bdb || "-";
  els.annotationInput.value = readAnnotation(state.selectedRef, index) || "";

  const entry = state.bdbData.entries[word.bdb];
  els.bdbEntry.textContent = entry
    ? `${entry.headword}\n\n${entry.glosses.join(", ")}\n\n${entry.definition}`
    : "No BDB entry found for this key in loaded dictionary file.";
}

function clearWordDetail() {
  els.emptyState.classList.remove("hidden");
  els.wordDetail.classList.add("hidden");
}

function saveAnnotation() {
  if (state.selectedWordIndex === null) {
    return;
  }

  const value = els.annotationInput.value.trim();
  writeAnnotation(state.selectedRef, state.selectedWordIndex, value);
  renderVerse();
  selectWord(state.selectedWordIndex);
}

function annotationKey(ref, index) {
  return `ann:${ref.book}:${ref.chapter}:${ref.verse}:${index}`;
}

function readAnnotation(ref, index) {
  return localStorage.getItem(annotationKey(ref, index));
}

function writeAnnotation(ref, index, value) {
  const key = annotationKey(ref, index);
  if (!value) {
    localStorage.removeItem(key);
    return;
  }
  localStorage.setItem(key, value);
}
