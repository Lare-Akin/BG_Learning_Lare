import streamlit as st
import pandas as pd
import altair as alt
from gtts import gTTS
import tempfile
import random
import os
from datetime import datetime
import time
import hashlib
from uuid import uuid4
import csv

# -----------------------------
# CONFIG - USING RELATIVE PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "bulgarian_reference.csv")
PROGRESS_PATH = os.path.join(BASE_DIR, "data", "quiz_progress.csv")
PHRASES_PATH = os.path.join(BASE_DIR, "data", "Bulgarian_Phrases.csv")
CONVO_PATH = os.path.join(BASE_DIR, "data", "Learn_From_Human_Conversation.csv")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# -----------------------------
# DATA LOADERS
# -----------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        for c in ["Category", "Bulgarian", "English"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df[df.get("Bulgarian", "") != ""].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


@st.cache_data
def load_phrases(path=PHRASES_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        df = df.dropna(axis=1, how="all")
        for c in ["Category", "Bulgarian", "English", "Pronunciation"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df[df.get("Bulgarian", "") != ""].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading phrases CSV: {e}")
        return pd.DataFrame()


@st.cache_data
def load_conversations(path=CONVO_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        col_map = {}
        for c in df.columns:
            c_low = c.lower()
            if "category" in c_low:
                col_map[c] = "Category"
            elif "bulgar" in c_low:
                col_map[c] = "Bulgarian"
            elif "pronun" in c_low or "pronunciation" in c_low:
                col_map[c] = "Pronunciation"
            elif "english" in c_low or "translation" in c_low:
                col_map[c] = "English"
        df = df.rename(columns=col_map)
        for c in ["Category", "Bulgarian", "Pronunciation", "English"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df[df.get("Bulgarian", "") != ""].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading conversations CSV: {e}")
        return pd.DataFrame()


df = load_data()
df_phrases = load_phrases()
df_convo = load_conversations()

# -----------------------------
# AUDIO HELPER
# -----------------------------
@st.cache_resource
def tts_audio(text, lang="bg"):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmpfile.name)
        return tmpfile.name
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return None

# -----------------------------
# SAVE PROGRESS (robust csv append)
# -----------------------------
def save_progress_to_csv(results, filepath=PROGRESS_PATH):
    try:
        if isinstance(results, dict):
            results = [results]
        if not results:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        CANONICAL_FIELDS = [
            "UserId", "SessionId", "ItemId", "Category", "English", "Bulgarian",
            "Chosen", "Correct", "ResponseTimeSec", "AttemptNumber", "QuestionType", "Timestamp"
        ]
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
            if write_header:
                writer.writeheader()
            for row in results:
                out = {k: row.get(k, "") for k in CANONICAL_FIELDS}
                writer.writerow(out)
    except Exception as e:
        st.error(f"Error saving progress (csv): {e}")

# -----------------------------
# CONTRIBUTION SYSTEM
# -----------------------------
def save_contribution(contribution_type, data):
    try:
        if contribution_type == "words":
            file_path = DATA_PATH
            expected_columns = ["Category", "Bulgarian", "English", "Transliteration", "Grammar_Notes"]
        elif contribution_type == "phrases":
            file_path = PHRASES_PATH
            expected_columns = ["Category", "Bulgarian", "Pronunciation", "English"]
        elif contribution_type == "conversations":
            file_path = CONVO_PATH
            expected_columns = ["Category", "Bulgarian", "Pronunciation", "English"]
        else:
            st.error("Invalid contribution type")
            return False

        new_row = {col: data.get(col, "") for col in expected_columns}

        try:
            existing_df = pd.read_csv(file_path, encoding="utf-8")
        except Exception:
            existing_df = pd.DataFrame(columns=expected_columns)

        new_df = pd.DataFrame([new_row])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(file_path, index=False, encoding="utf-8")

        if contribution_type == "words":
            load_data.clear()
        elif contribution_type == "phrases":
            load_phrases.clear()
        elif contribution_type == "conversations":
            load_conversations.clear()

        return True

    except Exception as e:
        st.error(f"Error saving contribution: {e}")
        return False


def contribution_section():
    with st.expander("🤝 Contribute Content", expanded=False):
        st.write("Help grow Laré's Bulgarian learning database by adding new words, phrases, or conversations!")

        contribution_type = st.selectbox(
            "What would you like to contribute?",
            ["Bulgarian Words", "Bulgarian Phrases", "Learn From Human Conversations"]
        )

        with st.form("contribution_form", clear_on_submit=True):
            st.subheader(f"Add New {contribution_type}")

            category = st.text_input("Category*", help="e.g., Food, Greetings, Travel, etc.")
            bulgarian = st.text_area("Bulgarian Text*", help="Enter the Bulgarian word/phrase")
            english = st.text_area("English Translation*", help="Enter the English translation")

            if contribution_type == "Bulgarian Words":
                transliteration = st.text_input("Transliteration", help="How to pronounce it (optional)")
                grammar_notes = st.text_area("Grammar Notes", help="Any grammatical notes (optional)")
            else:
                pronunciation = st.text_input("Pronunciation", help="How to pronounce it (optional)")

            contributor_name = st.text_input("Your Name (optional)", help="So we can thank you!")

            submitted = st.form_submit_button("Submit Contribution")

            if submitted:
                if not category or not bulgarian or not english:
                    st.error("Please fill in all required fields (*)")
                    return

                contribution_data = {
                    "Category": category.strip(),
                    "Bulgarian": bulgarian.strip(),
                    "English": english.strip()
                }

                if contribution_type == "Bulgarian Words":
                    contribution_data["Transliteration"] = transliteration.strip()
                    contribution_data["Grammar_Notes"] = grammar_notes.strip()
                    save_type = "words"
                else:
                    contribution_data["Pronunciation"] = pronunciation.strip()
                    save_type = "phrases" if contribution_type == "Bulgarian Phrases" else "conversations"

                if save_contribution(save_type, contribution_data):
                    st.success("🎉 Thank you for your contribution! The content has been added to the database.")

                    st.subheader("Preview of your contribution:")
                    if contribution_type == "Bulgarian Words":
                        lesson_card(
                            bulgarian=bulgarian,
                            pronunciation=transliteration,
                            english=english,
                            notes=grammar_notes
                        )
                    else:
                        lesson_card(
                            bulgarian=bulgarian,
                            pronunciation=pronunciation,
                            english=english,
                            notes=None
                        )

                    if contributor_name:
                        st.info(f"Thanks, {contributor_name}! 🙏")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
def init_session_state():
    defaults = {
        "score_history": [],
        "recent_quizzes": [],
        "accuracy_log": [],
        "category_scores": {},
        "quiz": [],
        "quiz_length": 0,
        "word_cat": None,
        "word_index_global": 0,
        "category_word_indices": {},
        "phrase_category_indices": {},
        "convo_category_indices": {},
        "user_id": "local_user",
        "session_id": None,
        "review_queue": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# -----------------------------
# SMALL UTILITIES
# -----------------------------
def make_item_id(bulgarian, english):
    key = f"{str(bulgarian)}||{str(english)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def ensure_itemid(df_progress):
    if df_progress is None or df_progress.empty:
        return df_progress
    if "ItemId" not in df_progress.columns:
        df_progress["ItemKey"] = (
            df_progress.get("Bulgarian", "").astype(str)
            + "||"
            + df_progress.get("English", "").astype(str)
        )
        df_progress["ItemId"] = df_progress["ItemKey"].apply(
            lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
        )
    return df_progress


def show_last_progress(filepath=PROGRESS_PATH, n=10):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            dfp = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            st.subheader("Recent progress rows")
            st.write(f"File: {filepath} — {len(dfp)} rows")
            st.dataframe(dfp.tail(n))
            return dfp
        except Exception as e:
            st.error(f"Unable to read progress CSV: {e}")
            return pd.DataFrame()
    else:
        st.info("No progress file found or file is empty.")
        return pd.DataFrame()


def compute_item_stats(progress_df):
    if progress_df is None or progress_df.empty:
        return pd.DataFrame()
    dfp = progress_df.copy()
    if "Timestamp" in dfp.columns:
        dfp["Timestamp"] = pd.to_datetime(dfp["Timestamp"], errors="coerce")
    dfp["Correct"] = pd.to_numeric(dfp.get("Correct", 0), errors="coerce").fillna(0).astype(int)
    dfp["ResponseTimeSec"] = pd.to_numeric(dfp.get("ResponseTimeSec", pd.NA), errors="coerce")
    dfp = ensure_itemid(dfp)
    agg = dfp.groupby("ItemId").agg(
        attempts=("Correct", "count"),
        corrects=("Correct", "sum"),
        avg_rt=("ResponseTimeSec", "mean"),
        last_seen=("Timestamp", "max"),
        sample_Bulgarian=("Bulgarian", "first"),
        sample_English=("English", "first"),
        sample_Category=("Category", "first"),
    ).reset_index()
    agg["error_rate"] = 1 - (agg["corrects"] / agg["attempts"])
    agg["difficulty_laplace"] = (agg["attempts"] - agg["corrects"] + 1) / (agg["attempts"] + 2)
    return agg

# -----------------------------
# BALANCED SAMPLER (predictive / difficulty-aware)
# -----------------------------
def sample_mixed_quiz(
    df_words,
    df_phrases,
    df_convos,
    progress_df,
    n=10,
    mix={"words": 0.5, "phrases": 0.3, "convos": 0.2},
    soften=0.9,
):
    if progress_df is None or progress_df.empty:
        pool = []
        if not df_words.empty:
            pool += df_words.dropna(subset=["Bulgarian", "English"]).sample(frac=1).to_dict("records")
        if not df_phrases.empty:
            pool += df_phrases.dropna(subset=["Bulgarian", "English"]).sample(frac=1).to_dict("records")
        if not df_convos.empty:
            pool += df_convos.dropna(subset=["Bulgarian", "English"]).sample(frac=1).to_dict("records")
        random.shuffle(pool)
        return pool[:n]

    stats = compute_item_stats(progress_df)
    difficulty = stats.set_index("ItemId")["difficulty_laplace"].to_dict() if not stats.empty else {}

    def weighted_sample(df_source, k):
        rows = df_source.dropna(subset=["Bulgarian", "English"]).copy()
        if rows.empty:
            return []
        rows["ItemId"] = rows.apply(
            lambda r: make_item_id(r["Bulgarian"], r["English"]), axis=1
        )
        rows["dscore"] = rows["ItemId"].map(difficulty).fillna(0.5)
        rows["weight"] = rows["dscore"] * soften + (1 - soften)
        rows["weight"] = rows["weight"].clip(0.01, 1.0)
        k = min(k, len(rows))
        chosen = rows.sample(n=k, weights="weight", replace=False)
        return chosen.to_dict("records")

    n_words = int(round(n * mix.get("words", 0)))
    n_phrases = int(round(n * mix.get("phrases", 0)))
    n_convos = n - n_words - n_phrases

    sampled = []
    if not df_words.empty and n_words > 0:
        sampled += weighted_sample(df_words, n_words)
    if not df_phrases.empty and n_phrases > 0:
        sampled += weighted_sample(df_phrases, n_phrases)
    if not df_convos.empty and n_convos > 0:
        sampled += weighted_sample(df_convos, n_convos)

    random.shuffle(sampled)
    return sampled[:n]

# -----------------------------
# UI: Title and navigation
# -----------------------------
st.title("Laré BG Learning")
st.write("Interactive learning for a bilingual person learning a new language")

# -----------------------------
# Helper: render a clean lesson card
# -----------------------------
def lesson_card(bulgarian, pronunciation, english, notes=None, audio_lang="bg"):
    st.markdown("---")
    st.markdown("#### 📝 Lesson card")
    colA, colB, colC = st.columns(3)
    colA.markdown("**🇧🇬 Bulgarian:**")
    colA.markdown(f"{bulgarian}")
    colB.markdown("**🔊 Pronunciation:**")
    colB.markdown(f"{pronunciation if pronunciation else '—'}")
    colC.markdown("**🌍 English:**")
    colC.markdown(f"{english}")

    audio_file = tts_audio(bulgarian, lang=audio_lang)
    if audio_file:
        st.audio(audio_file)

    if notes:
        st.info(f"📘 Notes: {notes}")

# -----------------------------
# Bulgarian Words Section
# -----------------------------
with st.expander("📚 Bulgarian Words (Lessons)"):
    if df.empty:
        st.warning("No word data available.")
    else:
        categories = sorted(df["Category"].dropna().unique()) if "Category" in df.columns else []
        if categories:
            chosen_cat = st.selectbox("Choose a category:", categories, key="word_cat_select")
            if st.session_state.word_cat != chosen_cat:
                st.session_state.word_cat = chosen_cat
                st.session_state.category_word_indices[chosen_cat] = 0
            subset = df[df["Category"] == chosen_cat]
            key_prefix = f"word_{chosen_cat.replace(' ', '_')}"
            word_list = subset["Bulgarian"].dropna().unique().tolist()
            if not word_list:
                st.warning("No words in this category.")
            else:
                idx = st.session_state.category_word_indices.get(chosen_cat, 0)
                col1, col2, col3 = st.columns([1, 2, 1])
                if col1.button("Previous", key=f"prev_{key_prefix}"):
                    idx = (idx - 1) % len(word_list)
                col2.write(f"**{idx+1} / {len(word_list)}**")
                if col3.button("Next", key=f"next_{key_prefix}"):
                    idx = (idx + 1) % len(word_list)
                st.session_state.category_word_indices[chosen_cat] = idx

                current_word = word_list[idx]
                row = subset[subset["Bulgarian"] == current_word].iloc[0]
                translit = row["Transliteration"] if "Transliteration" in row.index else ""
                notes = row["Grammar_Notes"] if "Grammar_Notes" in row.index else ""
                lesson_card(
                    bulgarian=row["Bulgarian"],
                    pronunciation=translit,
                    english=row["English"],
                    notes=notes,
                )

# -----------------------------
# Bulgarian Phrases Section
# -----------------------------
with st.expander("💬 Bulgarian Phrases"):
    if df_phrases.empty:
        st.warning("No phrase data available.")
    else:
        phrase_cats = sorted(df_phrases["Category"].dropna().unique()) if "Category" in df_phrases.columns else []
        if not phrase_cats:
            st.warning("No categories found in phrases data.")
        else:
            for cat in phrase_cats:
                with st.expander(f"{cat}"):
                    subset = df_phrases[df_phrases["Category"] == cat]
                    phrase_list = subset["Bulgarian"].dropna().unique().tolist()
                    if not phrase_list:
                        st.info("No phrases in this category.")
                        continue
                    if cat not in st.session_state.phrase_category_indices:
                        st.session_state.phrase_category_indices[cat] = 0
                    idx = st.session_state.phrase_category_indices[cat]
                    key_prefix = f"phrase_{cat.replace(' ', '_')}"
                    col1, col2, col3 = st.columns([1, 2, 1])
                    if col1.button("Previous", key=f"prev_{key_prefix}"):
                        idx = (idx - 1) % len(phrase_list)
                    col2.write(f"**{idx+1} / {len(phrase_list)}**")
                    if col3.button("Next", key=f"next_{key_prefix}"):
                        idx = (idx + 1) % len(phrase_list)
                    st.session_state.phrase_category_indices[cat] = idx

                    current_phrase = phrase_list[idx]
                    row = subset[subset["Bulgarian"] == current_phrase].iloc[0]
                    pron = row["Pronunciation"] if "Pronunciation" in row.index else ""
                    lesson_card(
                        bulgarian=row["Bulgarian"],
                        pronunciation=pron,
                        english=row["English"],
                        notes=None,
                    )

# -----------------------------
# Learn From Human Conversations Section
# -----------------------------
with st.expander("🗣️ Learn From Human Conversations"):
    if df_convo.empty:
        st.warning("No conversation data available. Check your CSV at CONVO_PATH.")
    else:
        convo_cats = sorted(df_convo["Category"].dropna().unique()) if "Category" in df_convo.columns else []
        if not convo_cats:
            st.warning("No categories found in the conversation CSV.")
        else:
            for cat in convo_cats:
                with st.expander(f"{cat}"):
                    subset = df_convo[df_convo["Category"] == cat]
                    convo_list = subset["Bulgarian"].dropna().unique().tolist()
                    if not convo_list:
                        st.info("No lines in this category.")
                        continue
                    if cat not in st.session_state.convo_category_indices:
                        st.session_state.convo_category_indices[cat] = 0
                    idx = st.session_state.convo_category_indices[cat]
                    key_prefix = f"convo_{cat.replace(' ', '_')}"
                    col1, col2, col3 = st.columns([1, 2, 1])
                    if col1.button("Previous", key=f"prev_{key_prefix}"):
                        idx = (idx - 1) % len(convo_list)
                    col2.write(f"**{idx+1} / {len(convo_list)}**")
                    if col3.button("Next", key=f"next_{key_prefix}"):
                        idx = (idx + 1) % len(convo_list)
                    st.session_state.convo_category_indices[cat] = idx

                    current_line = convo_list[idx]
                    row = subset[subset["Bulgarian"] == current_line].iloc[0]
                    pron = row["Pronunciation"] if "Pronunciation" in row.index else ""
                    eng = row["English"] if "English" in row.index else ""
                    lesson_card(
                        bulgarian=row["Bulgarian"],
                        pronunciation=pron,
                        english=eng,
                        notes=None,
                    )

# -----------------------------
# QUIZ SECTION (with proper saving)
# -----------------------------
with st.expander("🎯 Custom Quiz"):
    user_id = st.session_state.get("user_id", "local_user")
    if "session_id" not in st.session_state or st.session_state.get("session_id") is None:
        st.session_state["session_id"] = str(uuid4())
    session_id = st.session_state["session_id"]

    if df.empty or not set(["English", "Bulgarian"]).issubset(df.columns):
        st.warning("Quiz unavailable: required columns missing in reference data.")
    else:
        num_questions = st.slider("How many questions would you like?", 1, 20, 8)
        mix_choice = st.selectbox(
            "Quiz mix preset",
            ["Balanced (50/30/20)", "Quick Review (70/20/10)", "Phrases heavy (20/60/20)"],
        )
        if mix_choice == "Balanced (50/30/20)":
            mix = {"words": 0.5, "phrases": 0.3, "convos": 0.2}
        elif mix_choice == "Quick Review (70/20/10)":
            mix = {"words": 0.7, "phrases": 0.2, "convos": 0.1}
        else:
            mix = {"words": 0.2, "phrases": 0.6, "convos": 0.2}

        # Load progress for sampling (predictive)
        if os.path.exists(PROGRESS_PATH) and os.path.getsize(PROGRESS_PATH) > 0:
            progress_df_for_sampling = pd.read_csv(PROGRESS_PATH, encoding="utf-8", on_bad_lines="skip")
        else:
            progress_df_for_sampling = pd.DataFrame()

        # Review queue option (still uses predictive sampling)
        if st.session_state.review_queue:
            if st.button("Start Review Queue Quiz"):
                items_for_quiz = []
                combined_sources = pd.concat(
                    [df.assign(_src="words"), df_phrases.assign(_src="phrases"), df_convo.assign(_src="convos")],
                    ignore_index=True,
                )
                for iid in st.session_state.review_queue:
                    match = combined_sources[
                        combined_sources.apply(
                            lambda r: make_item_id(r.get("Bulgarian", ""), r.get("English", "")) == iid, axis=1
                        )
                    ]
                    if not match.empty:
                        items_for_quiz.append(match.iloc[0].to_dict())
                if len(items_for_quiz) < num_questions:
                    fill = sample_mixed_quiz(
                        df, df_phrases, df_convo, progress_df_for_sampling, n=num_questions - len(items_for_quiz), mix=mix
                    )
                    items_for_quiz += fill

                quiz_items = []
                combined_available = pd.concat([df, df_phrases, df_convo], ignore_index=True)
                for row in items_for_quiz[:num_questions]:
                    item_id = make_item_id(row.get("Bulgarian", ""), row.get("English", ""))
                    distractors = combined_available[combined_available["Bulgarian"] != row.get("Bulgarian", "")][
                        "Bulgarian"
                    ].dropna()
                    distracts = distractors.sample(min(3, len(distractors))).tolist() if len(distractors) > 0 else []
                    options = distracts + [row.get("Bulgarian", "")]
                    random.shuffle(options)
                    quiz_items.append(
                        {
                            "ItemId": item_id,
                            "English": row.get("English", ""),
                            "Bulgarian": row.get("Bulgarian", ""),
                            "Transliteration": row.get("Transliteration", "")
                            if "Transliteration" in row
                            else row.get("Pronunciation", ""),
                            "Options": options,
                            "Answered": False,
                            "Correct": None,
                            "Chosen": None,
                            "AttemptNumber": 0,
                            "question_shown_ts": None,
                            "question_answered_ts": None,
                        }
                    )
                st.session_state.quiz = quiz_items
                st.session_state.quiz_length = num_questions

        # Start new quiz (predictive or fresh)
        if st.button("Start New Quiz") or st.session_state.get("quiz_length") != num_questions:
            sampled = sample_mixed_quiz(df, df_phrases, df_convo, progress_df_for_sampling, n=num_questions, mix=mix)
            quiz_items = []
            combined_available = pd.concat([df, df_phrases, df_convo], ignore_index=True)
            for row in sampled:
                item_id = make_item_id(row.get("Bulgarian", ""), row.get("English", ""))
                distractors = combined_available[combined_available["Bulgarian"] != row.get("Bulgarian", "")][
                    "Bulgarian"
                ].dropna()
                distracts = distractors.sample(min(3, len(distractors))).tolist() if len(distractors) > 0 else []
                options = distracts + [row.get("Bulgarian", "")]
                random.shuffle(options)
                quiz_items.append(
                    {
                        "ItemId": item_id,
                        "English": row.get("English", ""),
                        "Bulgarian": row.get("Bulgarian", ""),
                        "Transliteration": row.get("Transliteration", "")
                        if "Transliteration" in row
                        else row.get("Pronunciation", ""),
                        "Options": options,
                        "Answered": False,
                        "Correct": None,
                        "Chosen": None,
                        "AttemptNumber": 0,
                        "question_shown_ts": None,
                        "question_answered_ts": None,
                    }
                )
            st.session_state.quiz = quiz_items
            st.session_state.quiz_length = num_questions

        score = 0
        for i, item in enumerate(st.session_state.quiz):
            st.subheader(f"Question {i+1}")
            st.write(f"How do you say **{item['English']}** in Bulgarian?")

            if st.session_state.quiz[i]["question_shown_ts"] is None:
                st.session_state.quiz[i]["question_shown_ts"] = time.time()

            if item.get("Transliteration"):
                st.caption(f"Pronunciation hint: {item['Transliteration']}")

            selected = st.radio("Choose:", item["Options"], key=f"q_{i}")

            if st.button("Check Answer", key=f"check_{i}") and not item["Answered"]:
                st.session_state.quiz[i]["AttemptNumber"] = item.get("AttemptNumber", 0) + 1
                st.session_state.quiz[i]["question_answered_ts"] = time.time()
                response_time = (
                    st.session_state.quiz[i]["question_answered_ts"]
                    - st.session_state.quiz[i]["question_shown_ts"]
                )
                chosen = selected
                correct = chosen == item["Bulgarian"]
                st.session_state.quiz[i]["Chosen"] = chosen
                st.session_state.quiz[i]["Answered"] = True
                st.session_state.quiz[i]["Correct"] = correct

                category_val = ""
                try:
                    combined = pd.concat([df, df_phrases, df_convo], ignore_index=True)
                    match_df = combined.loc[
                        (combined["Bulgarian"] == item["Bulgarian"])
                        & (combined["English"] == item["English"])
                    ]
                    if not match_df.empty and "Category" in match_df.columns:
                        category_val = str(match_df["Category"].iloc[0])
                except Exception:
                    category_val = ""

                # ✅ SAVE PROGRESS HERE
                result = {
                    "UserId": user_id,
                    "SessionId": session_id,
                    "ItemId": item["ItemId"],
                    "Category": category_val,
                    "English": item["English"],
                    "Bulgarian": item["Bulgarian"],
                    "Chosen": chosen,
                    "Correct": bool(correct),
                    "ResponseTimeSec": round(response_time, 2),
                    "AttemptNumber": st.session_state.quiz[i]["AttemptNumber"],
                    "QuestionType": "MCQ",
                    "Timestamp": datetime.now().isoformat(),
                }
                save_progress_to_csv(result)

                st.session_state.recent_quizzes.append(
                    {
                        "Session": len(st.session_state.accuracy_log) + 1,
                        "English": item["English"],
                        "Correct": item["Bulgarian"],
                        "Chosen": chosen,
                        "Correct?": correct,
                    }
                )
                st.session_state.score_history.append(1 if correct else 0)

                if category_val:
                    st.session_state.category_scores.setdefault(
                        category_val, {"correct": 0, "total": 0}
                    )
                    st.session_state.category_scores[category_val]["total"] += 1
                    if correct:
                        st.session_state.category_scores[category_val]["correct"] += 1

                if correct:
                    st.success("Correct! 🎉")
                    audio_file = tts_audio(item.get("Bulgarian", ""))
                    if audio_file:
                        st.audio(audio_file)
                else:
                    translit = item.get("Transliteration", "")
                    st.error(f"Not quite. The correct answer is {item['Bulgarian']} ({translit})")
            elif item["Answered"]:
                if item["Correct"]:
                    st.success("✅ Already answered correctly")
                else:
                    st.error(f"❌ Already answered. Correct: {item['Bulgarian']}")

            if item.get("Correct"):
                score += 1

        if st.session_state.get("quiz"):
            st.markdown(f"### ✅ Your Score: {score} out of {len(st.session_state.quiz)}")
            session_accuracy = round((score / len(st.session_state.quiz)) * 100, 1)
            st.session_state.accuracy_log.append(
                {"Session": len(st.session_state.accuracy_log) + 1, "Accuracy": session_accuracy}
            )

# -----------------------------
# PROGRESS DASHBOARD (Altair + predictive stats)
# -----------------------------
with st.expander("📊 Progress Dashboard"):
    progress_df = pd.DataFrame()
    if os.path.exists(PROGRESS_PATH) and os.path.getsize(PROGRESS_PATH) > 0:
        try:
            progress_df = pd.read_csv(PROGRESS_PATH, encoding="utf-8", on_bad_lines="skip")
            if "Timestamp" in progress_df.columns:
                progress_df["Timestamp"] = pd.to_datetime(progress_df["Timestamp"], errors="coerce")
        except Exception as e:
            st.error(f"Error loading progress data: {e}")

    if not progress_df.empty:
        total_quizzes = len(progress_df)
        total_correct = progress_df["Correct"].sum() if "Correct" in progress_df.columns else 0
        accuracy = round((total_correct / total_quizzes) * 100, 1) if total_quizzes else 0

        category_performance = {}
        if "Category" in progress_df.columns and "Correct" in progress_df.columns:
            for category in progress_df["Category"].unique():
                if category and pd.notna(category):
                    cat_data = progress_df[progress_df["Category"] == category]
                    cat_total = len(cat_data)
                    cat_correct = cat_data["Correct"].sum()
                    cat_accuracy = round((cat_correct / cat_total) * 100, 1) if cat_total else 0
                    category_performance[category] = cat_accuracy
    else:
        total_quizzes = len(st.session_state.score_history)
        total_correct = sum(st.session_state.score_history)
        accuracy = round((total_correct / total_quizzes) * 100, 1) if total_quizzes else 0
        category_performance = {
            cat: round((data["correct"] / data["total"]) * 100, 1)
            for cat, data in st.session_state.category_scores.items()
            if data["total"] > 0
        }

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions Attempted", total_quizzes)
    col2.metric("Correct Answers", total_correct)
    col3.metric("Accuracy", f"{accuracy}%")

    # Historical accuracy (Altair or line_chart)
    if not progress_df.empty and "Timestamp" in progress_df.columns and "Correct" in progress_df.columns:
        st.subheader("📈 Historical Accuracy Over Time")
        progress_df["Date"] = progress_df["Timestamp"].dt.date
        daily_accuracy = progress_df.groupby("Date")["Correct"].mean() * 100
        daily_accuracy = daily_accuracy.reset_index()
        daily_accuracy.columns = ["Date", "Accuracy"]
        if not daily_accuracy.empty:
            chart = (
                alt.Chart(daily_accuracy)
                .mark_line(point=True)
                .encode(
                    x="Date:T",
                    y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["Date", "Accuracy"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data for historical accuracy chart.")
    elif st.session_state.accuracy_log:
        st.subheader("📈 Accuracy Over Sessions")
        acc_df = pd.DataFrame(st.session_state.accuracy_log)
        st.line_chart(acc_df.set_index("Session"))
    else:
        st.info("Complete a quiz to see accuracy trends.")

    if category_performance:
        st.subheader("🧠 Category Mastery")
        mastery_df = pd.DataFrame.from_dict(category_performance, orient="index", columns=["Mastery %"])
        st.bar_chart(mastery_df)
    else:
        st.info("Complete quizzes to see category mastery.")

    if not progress_df.empty:
        mistakes = progress_df[progress_df["Correct"] == False].tail(5)
        if not mistakes.empty:
            st.subheader("🔁 Recent Mistakes")
            for _, mistake in mistakes.iterrows():
                st.markdown(
                    f"**English:** {mistake.get('English', '')}  \n"
                    f"**Your Answer:** {mistake.get('Chosen', '')}  \n"
                    f"**Correct Answer:** {mistake.get('Bulgarian', '')}"
                )
                audio_file = tts_audio(mistake.get("Bulgarian", ""))
                if audio_file:
                    st.audio(audio_file)
        else:
            st.info("No recent mistakes to review. Great job!")
    else:
        st.info("Complete a quiz to see mistake analysis.")

# -----------------------------
# Analysis of improvement areas (predictive; Altair scatter & bars)
# -----------------------------
with st.expander("🔍 Analysis of Improvement Areas"):
    progress_df = show_last_progress(PROGRESS_PATH, n=500)

    if progress_df is not None and not progress_df.empty:
        items = compute_item_stats(progress_df)

        if not items.empty:
            st.subheader("📋 Top 15 Hardest Items")
            hard = items.sort_values("difficulty_laplace", ascending=False).head(15)

            for _, row in hard.iterrows():
                cols = st.columns([3, 3, 1])
                cols[0].write(f"**{row['sample_Bulgarian']}** — {row['sample_English']}")
                cols[1].write(
                    f"Category: {row['sample_Category']} • Attempts: {int(row['attempts'])} • "
                    f"Error: {row['error_rate']:.2f}"
                )
                add_key = f"add_{row['ItemId']}"
                if cols[2].button("Add to review", key=add_key):
                    if row["ItemId"] not in st.session_state.review_queue:
                        st.session_state.review_queue.append(row["ItemId"])
                        st.success("Added to review queue")
                    else:
                        st.info("Already in review queue")

            st.subheader("📊 Category Mastery (from item stats)")
            cat_stats = items.groupby("sample_Category").agg(
                Items=("attempts", "count"),
                Avg_Error_Rate=("error_rate", "mean"),
                Total_Correct=("corrects", "sum"),
            ).reset_index()
            cat_stats["Mastery %"] = 100 - (cat_stats["Avg_Error_Rate"] * 100)
            st.dataframe(cat_stats)

            st.subheader("⏰ Recency vs Error Rate")
            items["days_since"] = (pd.Timestamp.now() - items["last_seen"]).dt.days

            if not items.empty and "days_since" in items.columns and "error_rate" in items.columns:
                chart = (
                    alt.Chart(items)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X("days_since:Q", title="Days Since Last Seen"),
                        y=alt.Y("error_rate:Q", title="Error Rate", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("sample_Category:N", title="Category"),
                        tooltip=[
                            "sample_Bulgarian",
                            "sample_English",
                            "attempts",
                            "error_rate",
                            "avg_rt",
                            "last_seen",
                        ],
                    )
                    .properties(height=400, title="Error Rate vs Days Since Last Seen")
                )
                st.altair_chart(chart, use_container_width=True)

            st.subheader("⏱️ Response Time Analysis")
            rt_stats = items.groupby("sample_Category")["avg_rt"].median().reset_index()
            rt_stats.columns = ["Category", "Median Response Time (s)"]
            st.bar_chart(rt_stats.set_index("Category"))
        else:
            st.info("No item statistics available yet.")
    else:
        st.info("No progress data to analyze yet. Complete some quizzes first.")

    st.markdown("### 📝 Review Queue")
    if st.session_state.review_queue:
        queue_items = []
        items_stats = compute_item_stats(progress_df) if not progress_df.empty else pd.DataFrame()

        for item_id in st.session_state.review_queue:
            if not items_stats.empty:
                item_data = items_stats[items_stats["ItemId"] == item_id]
                if not item_data.empty:
                    queue_items.append(
                        {
                            "ItemId": item_id,
                            "Bulgarian": item_data["sample_Bulgarian"].iloc[0],
                            "English": item_data["sample_English"].iloc[0],
                            "Error Rate": f"{item_data['error_rate'].iloc[0]:.2f}",
                        }
                    )
                else:
                    queue_items.append(
                        {
                            "ItemId": item_id,
                            "Bulgarian": "Not found",
                            "English": "Not found",
                            "Error Rate": "N/A",
                        }
                    )
            else:
                queue_items.append(
                    {
                        "ItemId": item_id,
                        "Bulgarian": "Unknown",
                        "English": "Unknown",
                        "Error Rate": "N/A",
                    }
                )

        st.dataframe(pd.DataFrame(queue_items))

        col1, col2 = st.columns(2)
        if col1.button("Clear Review Queue"):
            st.session_state.review_queue = []
            st.success("Review queue cleared!")
        if col2.button("Start Review Quiz"):
            st.rerun()
    else:
        st.info("Review queue is empty. Add items from the 'Top 15 Hardest Items' list.")

# -----------------------------
# CONTRIBUTION SECTION
# -----------------------------
contribution_section()
